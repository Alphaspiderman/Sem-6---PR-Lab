import numpy as np


def train_hopfield(patterns):
    w = sum(np.outer(p, p) for p in patterns)
    np.fill_diagonal(w, 0)
    return w

def recall_pattern(weights, inp, max_iter=10):
    out = np.array(inp)
    for _ in range(max_iter):
        out = np.sign(weights @ out)
    return out


orignal = [-1, 1, -1, -1, -1, -1, -1, 1, -1, 1]
weights = train_hopfield([orignal])
noisy = [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1]
print(f"Original Pattern:  {orignal}")
print(f"Noisy Pattern:     {noisy}")
print(f"Recovered Pattern: {recall_pattern(weights, noisy)}")
