from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch


class Exp_Basic(ABC):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        use_cuda = torch.cuda.is_available() and self.args.use_gpu
        if use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, setting: str):
        raise NotImplementedError

    @abstractmethod
    def test(self, setting: str):
        raise NotImplementedError

