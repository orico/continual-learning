"""Fixed training choices for the Split MNIST toy.

These stay constant across methods so a difference in the plots comes from the
method, not from a different optimizer budget.
"""

HIDDEN = 256
EPOCHS = 2
LR = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0
BUFFER_CAPACITY = 500
EWC_LAMBDA = 1000.0
LWF_ALPHA = 1.0
LWF_TEMPERATURE = 2.0
FISHER_SAMPLES = 256
SEEDS = (0, 1, 2)
BUFFER_SWEEP = (50, 200, 500, 1000)
EWC_LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 1_000_000.0)
N_TASKS = 5
CLASS_OUTPUTS = 10
