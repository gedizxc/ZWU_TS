import os
import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")
