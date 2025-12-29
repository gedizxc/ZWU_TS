import numpy as np


class StandardScalerPerChannel:
    """Per-channel mean/std fit on training portion."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray):
        self.mean_ = data.mean(axis=0, keepdims=True)
        self.std_ = data.std(axis=0, keepdims=True)
        self.std_ = np.maximum(self.std_, self.eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
