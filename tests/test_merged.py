"""The merged method adds both penalties, then projects that step with plain memory loss."""

import copy
import unittest

import torch
import torch.nn.functional as F

from cl_toy.buffer import ClassBalancedBuffer
from cl_toy.methods import (
    apply_agem,
    clone_frozen,
    distillation_loss,
    ewc_penalty,
    pieces_for,
    snapshot_params,
)
from cl_toy.model import MLP, predict_logits
from cl_toy.train import _new_optimizer, _update, current_loss, memory_loss


def _filled_model() -> MLP:
    model = MLP(n_outputs=2, n_heads=1, hidden=4, in_dim=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    return model


def _batch(label: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.ones(4, 4)
    labels = torch.full((4,), label, dtype=torch.long)
    task_ids = torch.zeros(4, dtype=torch.long)
    return images, labels, task_ids


def _buffer(images: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor) -> ClassBalancedBuffer:
    buffer = ClassBalancedBuffer(capacity=8, seed=0)
    digits = torch.ones(images.shape[0], dtype=torch.long)
    buffer.add_many(images, labels, task_ids, digits)
    return buffer


def _same(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    return all(
        torch.allclose(a, b, rtol=1e-5, atol=1e-6)
        for a, b in zip(left.parameters(), right.parameters())
    )


class MergedLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _filled_model()
        self.star = snapshot_params(self.model)
        self.teacher = clone_frozen(self.model)
        with torch.no_grad():
            self.teacher.head.bias[0] = 1.0
            self.model.fc1.weight[0, 0] = 0.8
        self.fisher = {
            name: torch.ones_like(parameter) for name, parameter in self.model.named_parameters()
        }
        self.anchors = [(self.star, self.fisher, 1.0)]
        self.images, self.labels, self.task_ids = _batch(0)
        self.memory_images, self.memory_labels, self.memory_task_ids = _batch(1)
        self.alpha = 1.0
        self.temperature = 2.0
        self.task_index = 1

    def _current(self, method: str, model: MLP | None = None, anchors: list | None = None, teacher: MLP | None = None) -> torch.Tensor:
        return current_loss(
            self.model if model is None else model,
            self.images,
            self.labels,
            self.task_ids,
            "domain",
            method,
            self.anchors if anchors is None else anchors,
            self.teacher if teacher is None else teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )

    def _memory(self, anchors: list | None = None, teacher: MLP | None = None) -> torch.Tensor:
        return memory_loss(
            self.model,
            self.memory_images,
            self.memory_labels,
            self.memory_task_ids,
            "domain",
            "merged",
            self.anchors if anchors is None else anchors,
            self.teacher if teacher is None else teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            pieces_for("not-a-method")

    def test_merged_loss_is_cross_entropy_plus_both_penalties(self) -> None:
        plain = F.cross_entropy(
            predict_logits(self.model, self.images, self.task_ids, "domain", "merged"),
            self.labels,
        )
        penalty = ewc_penalty(self.model, self.anchors)
        distill = self.alpha * distillation_loss(
            self.model, self.teacher, self.images, self.task_index, self.temperature
        )
        self.assertGreater(float(penalty.detach()), 0.0)
        self.assertGreater(float(distill.detach()), 0.0)
        combined = self._current("merged")
        self.assertTrue(torch.allclose(combined, plain + penalty + distill, rtol=1e-5, atol=1e-6))

    def test_memory_loss_ignores_the_snapshot_and_the_teacher(self) -> None:
        plain = F.cross_entropy(
            predict_logits(self.model, self.memory_images, self.memory_task_ids, "domain", "merged"),
            self.memory_labels,
        )
        full = self._memory()
        bare = self._memory(anchors=[], teacher=None)
        self.assertTrue(torch.allclose(full, plain, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(full, bare, rtol=1e-5, atol=1e-6))

    def test_agem_loss_stays_cross_entropy_and_its_step_matches_the_projection(self) -> None:
        plain = F.cross_entropy(
            predict_logits(self.model, self.images, self.task_ids, "domain", "agem"),
            self.labels,
        )
        self.assertTrue(torch.allclose(self._current("agem"), plain, rtol=1e-5, atol=1e-6))
        stepped = self._step("agem")
        projected = self._project("agem")
        self.assertTrue(_same(stepped, projected))

    def test_merged_step_matches_projection_of_the_combined_loss(self) -> None:
        stepped = self._step("merged")
        projected = self._project("merged")
        plain = self._plain_step("merged")
        self.assertTrue(_same(stepped, projected))
        self.assertFalse(_same(stepped, plain))

    def test_ewc_and_lwf_add_only_their_own_term_and_take_a_plain_step(self) -> None:
        plain = F.cross_entropy(
            predict_logits(self.model, self.images, self.task_ids, "domain", "ewc"),
            self.labels,
        )
        penalty = ewc_penalty(self.model, self.anchors)
        distill = self.alpha * distillation_loss(
            self.model, self.teacher, self.images, self.task_index, self.temperature
        )
        ewc = self._current("ewc")
        lwf = self._current("lwf")
        self.assertGreater(float(penalty.detach()), 0.0)
        self.assertGreater(float(distill.detach()), 0.0)
        self.assertTrue(torch.allclose(ewc, plain + penalty, rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.allclose(ewc, plain + penalty + distill, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(lwf, plain + distill, rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.allclose(lwf, plain + penalty + distill, rtol=1e-5, atol=1e-6))
        self.assertTrue(_same(self._step("ewc"), self._plain_step("ewc")))
        self.assertTrue(_same(self._step("lwf"), self._plain_step("lwf")))

    def _copy(self) -> tuple[MLP, torch.optim.Optimizer]:
        model = copy.deepcopy(self.model)
        optimizer = _new_optimizer(model, lr=0.05, momentum=0.9)
        for parameter in model.parameters():
            optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 0.01)
        return model, optimizer

    def _step(self, method: str) -> MLP:
        model, optimizer = self._copy()
        _update(
            model,
            optimizer,
            self.images,
            self.labels,
            self.task_ids,
            "domain",
            method,
            _buffer(self.memory_images, self.memory_labels, self.memory_task_ids),
            self.anchors,
            self.teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )
        return model

    def _project(self, method: str) -> MLP:
        model, optimizer = self._copy()
        model.train()
        loss_current = current_loss(
            model,
            self.images,
            self.labels,
            self.task_ids,
            "domain",
            method,
            self.anchors,
            self.teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )
        drawn = _buffer(self.memory_images, self.memory_labels, self.memory_task_ids).sample(self.images.size(0))
        assert drawn is not None
        ref_images, ref_labels, ref_task_ids = drawn
        loss_reference = memory_loss(
            model,
            ref_images,
            ref_labels,
            ref_task_ids,
            "domain",
            method,
            self.anchors,
            self.teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )
        apply_agem(model, optimizer, loss_current, loss_reference)
        return model

    def _plain_step(self, method: str) -> MLP:
        model, optimizer = self._copy()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = current_loss(
            model,
            self.images,
            self.labels,
            self.task_ids,
            "domain",
            method,
            self.anchors,
            self.teacher,
            self.task_index,
            self.alpha,
            self.temperature,
        )
        loss.backward()
        optimizer.step()
        return model


if __name__ == "__main__":
    unittest.main()
