"""Class-balanced reservoir of past examples.

Balance is by the original digit, not by the scenario label. In the
task-incremental setting several tasks share the local labels 0 and 1, and
balancing those labels would let a later pair evict an earlier digit.
"""

from __future__ import annotations

import random

import torch


class ClassBalancedBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.store: dict[int, list[tuple[torch.Tensor, int, int, int]]] = {}
        self.seen: dict[int, int] = {}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.store.values())

    def digit_counts(self) -> dict[int, int]:
        return {digit: len(bucket) for digit, bucket in self.store.items() if bucket}

    def add(self, image: torch.Tensor, label: int, task_id: int, digit: int) -> None:
        digit = int(digit)
        self.seen[digit] = self.seen.get(digit, 0) + 1
        if digit not in self.store:
            self.store[digit] = []
            self._trim_to_quotas()
        quota = self._quota(digit)
        if quota <= 0:
            return
        item = (
            image.detach().float().clone(),
            int(label),
            int(task_id),
            digit,
        )
        bucket = self.store[digit]
        if len(bucket) < quota:
            bucket.append(item)
            return
        # Standard reservoir: each example seen so far for this digit is equally
        # likely to occupy one of the quota slots.
        draw = self.rng.randrange(self.seen[digit])
        if draw < quota:
            bucket[draw] = item

    def add_many(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
        digits: torch.Tensor,
    ) -> None:
        for index in range(images.shape[0]):
            self.add(images[index], int(labels[index]), int(task_ids[index]), int(digits[index]))

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        items = [item for bucket in self.store.values() for item in bucket]
        if not items:
            return None
        n = min(n, len(items))
        chosen = self.rng.sample(items, n)
        images = torch.stack([item[0] for item in chosen])
        labels = torch.tensor([item[1] for item in chosen], dtype=torch.long)
        task_ids = torch.tensor([item[2] for item in chosen], dtype=torch.long)
        return images, labels, task_ids

    def _quota(self, digit: int) -> int:
        digits = sorted(self.store)
        if not digits or digit not in self.store:
            return 0
        base = self.capacity // len(digits)
        extra = self.capacity % len(digits)
        rank = digits.index(digit)
        return base + (1 if rank < extra else 0)

    def _trim_to_quotas(self) -> None:
        for digit in list(self.store):
            quota = self._quota(digit)
            bucket = self.store[digit]
            if len(bucket) <= quota:
                continue
            if quota <= 0:
                self.store[digit] = []
                continue
            keep = self.rng.sample(range(len(bucket)), quota)
            self.store[digit] = [bucket[index] for index in keep]
