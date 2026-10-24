# Split MNIST continual-learning results

## Setup

Each run trains a 784-256-256 multilayer perceptron with SGD (learning rate 0.01, momentum 0.9, no weight decay) for 2 epochs on each of five Split MNIST tasks: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9). Task-incremental evaluation chooses between the two digits of a known pair. Domain-incremental evaluation asks even versus odd with no task id. Class-incremental evaluation names the digit among all ten with no task id. The main comparison uses seeds [0, 1, 2]. Replay stores 500 examples. EWC uses lambda 1000. Accuracy in the saved metrics is a fraction from 0 to 1; the figures and this note use percents. The EWC Fisher is divided by its mean and capped so a large penalty pins weights instead of making SGD diverge. A-GEM projects the momentum update, because projecting only the latest gradient leaves the old velocity free to increase loss on stored examples.

## The picture

On this network, task-incremental naive training barely forgets (98.4% ± 0.1), because each pair keeps its own output head. Domain-incremental naive training, even versus odd with a shared head, ends at 60.9% ± 0.9; replay brings that to 93.4% ± 0.4. Class-incremental naive training is the collapse: final accuracy 19.5% ± 0.1, against joint training at 96.4% ± 0.2. Replay reaches 87.8% ± 0.0. A-GEM, which only constrains the step with stored gradients, reaches 65.3% ± 1.7. EWC at the default penalty stays near the naive result (21.7% ± 3.8); learning without forgetting, which stores no old images, reaches 26.0% ± 1.3. A frozen expert per task scores 98.9% ± 0.1 when it is handed the true task id, and 19.6% ± 0.0 when a sequentially trained router has to guess that id.

## What the runs showed

Task-incremental: Naive accuracy 98.4% ± 0.1, forgetting 1.1% ± 0.1; Replay accuracy 98.5% ± 0.1, forgetting 0.6% ± 0.1; EWC accuracy 97.7% ± 0.3, forgetting -0.0% ± 0.0; LwF accuracy 99.2% ± 0.0, forgetting 0.1% ± 0.0; A-GEM accuracy 98.7% ± 0.1, forgetting 0.7% ± 0.1; Experts (oracle id) accuracy 99.1% ± 0.1, forgetting 0.0% ± 0.0; Joint accuracy 99.3% ± 0.1.

Domain-incremental: Naive accuracy 60.9% ± 0.9, forgetting 47.6% ± 1.2; Replay accuracy 93.4% ± 0.4, forgetting 6.9% ± 0.5; EWC accuracy 70.6% ± 2.0, forgetting 21.1% ± 1.7; LwF accuracy 75.2% ± 0.4, forgetting 16.2% ± 0.2; A-GEM accuracy 84.3% ± 0.9, forgetting 18.7% ± 1.1; Experts (oracle id) accuracy 99.1% ± 0.1, forgetting 0.0% ± 0.0; Experts (predicted id) accuracy 56.7% ± 0.4, forgetting 53.0% ± 0.5; Joint accuracy 98.2% ± 0.0.

Class-incremental: Naive accuracy 19.5% ± 0.1, forgetting 99.1% ± 0.0; Replay accuracy 87.8% ± 0.0, forgetting 12.4% ± 0.2; EWC accuracy 21.7% ± 3.8, forgetting 95.6% ± 4.5; LwF accuracy 26.0% ± 1.3, forgetting 52.8% ± 2.1; A-GEM accuracy 65.3% ± 1.7, forgetting 42.1% ± 2.1; Experts (oracle id) accuracy 98.9% ± 0.1, forgetting 0.0% ± 0.0; Experts (predicted id) accuracy 19.6% ± 0.0, forgetting 99.2% ± 0.1; Joint accuracy 96.4% ± 0.2.

Parameter counts, identical across seeds:

- Task-incremental, Naive: 269,322 parameters.
- Task-incremental, Joint: 269,322 parameters.
- Task-incremental, Replay: 269,322 parameters.
- Task-incremental, EWC: 269,322 parameters.
- Task-incremental, LwF: 269,322 parameters.
- Task-incremental, A-GEM: 269,322 parameters.
- Task-incremental, Experts (oracle id): 1,336,330 parameters.
- Domain-incremental, Naive: 267,266 parameters.
- Domain-incremental, Joint: 267,266 parameters.
- Domain-incremental, Replay: 267,266 parameters.
- Domain-incremental, EWC: 267,266 parameters.
- Domain-incremental, LwF: 267,266 parameters.
- Domain-incremental, A-GEM: 267,266 parameters.
- Domain-incremental, Experts (oracle id): 1,336,330 parameters.
- Domain-incremental, Experts (predicted id): 1,604,367 parameters.
- Class-incremental, Naive: 269,322 parameters.
- Class-incremental, Joint: 269,322 parameters.
- Class-incremental, Replay: 269,322 parameters.
- Class-incremental, EWC: 269,322 parameters.
- Class-incremental, LwF: 269,322 parameters.
- Class-incremental, A-GEM: 269,322 parameters.
- Class-incremental, Experts (oracle id): 1,346,610 parameters.
- Class-incremental, Experts (predicted id): 1,614,647 parameters.

## Checks against the expected pattern

- Tasks are actually learned: held. Naive task-incremental accuracy just after each task ranges from 98.1% to 100.0%.
- Naive class-incremental training forgets: held. Mean just-learned accuracy is 98.8%; final average accuracy is 19.5% ± 0.1. The drop is 79.3 points.
- Task-incremental naive ends higher than class-incremental naive: held. Task-incremental 98.4% ± 0.1 versus class-incremental 19.5% ± 0.1 (gap 78.8 points, seed spread 0.1 points).
- Domain-incremental naive is reported as measured: held. Final average accuracy 60.9% ± 0.9.
- Replay beats naive on class-incremental accuracy and forgetting: held. Accuracy 87.8% ± 0.0 versus 19.5% ± 0.1 (gap 68.3 points, spread 0.1 points). Forgetting 12.4% ± 0.2 versus 99.1% ± 0.0 (naive is higher by 86.6 points, spread 0.2 points).
- Joint training ends above naive class-incremental training: held. Joint 96.4% ± 0.2 versus naive 19.5% ± 0.1 (gap 76.8 points, spread 0.2 points).
- Task 1 falls sharply early in task 2: held. Class-incremental naive accuracy on task 1 starts at 99.9% and is 0.0% after 19 steps of task 2 (drop 99.9 points). At the end of task 2 it is 0.0%.
- Some EWC lambda beats naive on task-incremental accuracy: held. Best lambda 1 reaches 98.8%; naive on the same seed is 98.3%.
- A large EWC lambda holds the first task up and the last task down: held. Lambda 1: task 1 100.0%, last task 97.5%. Lambda 1e+06: task 1 99.9%, last task 90.9%.
- A tiny EWC lambda behaves like naive training: held. Lambda 1 average accuracy 98.8% versus naive 98.3% (absolute gap 0.5 points).
- Class-incremental EWC is reported at its real effect: held. EWC 21.7% ± 3.8 versus naive 19.5% ± 0.1 at lambda 1000.
- A-GEM beats naive on class-incremental accuracy: held. A-GEM 65.3% ± 1.7 versus naive 19.5% ± 0.1 (gap 45.8 points, spread 1.7 points).
- Learning without Forgetting is reported at its real effect: held. Class-incremental LwF 26.0% ± 1.3 versus naive 19.5% ± 0.1. Task-incremental LwF 99.2% ± 0.0 versus naive 98.4% ± 0.1.
- Oracle experts on task-incremental learning almost do not forget: held. Average forgetting 0.0% ± 0.0.
- Predicted routing on class-incremental learning lands below the oracle: held. Oracle experts 98.9% ± 0.1 versus predicted routing 19.6% ± 0.0 (gap 79.3 points, spread 0.1 points).
- A larger replay buffer is closer to joint training: held. size 50: 59.6%; size 200: 80.2%; size 500: 87.7%; size 1000: 90.2%. Joint is 96.4% ± 0.2. Monotonic within 2 points: True. Largest buffer closer to joint: True.
- A single pass still forgets, and replay still helps: held. One-epoch class-incremental naive final accuracy 19.3% (drop from the diagonal 79.1 points). Replay finishes at 86.3%.

## How to read a partial result

A method works to some degree on a metric when its mean beats naive training by more than the seed spread. Elastic weight consolidation and learning without forgetting store no old images, so a small class-incremental gain is still the measured effect. Oracle experts are shown the true task id in every setting, including class-incremental learning, where that id is not part of the problem. Predicted routing is the version that has to discover the task. Joint training sees every task in a shuffled mix for the same number of epochs and is the ceiling when keeping all of the data is allowed.

