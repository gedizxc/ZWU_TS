from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.model import SimpleInformer
from utils.visualize import export_batch_correlation_images, export_batch_patch_correlation_videos


class Exp_Informer(Exp_Basic):
    def _build_model(self):
        # enc_in/c_out can be inferred from dataset later; keep args values for now.
        return SimpleInformer(
            enc_in=self.args.enc_in,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            pred_len=self.args.pred_len,
            c_out=self.args.c_out,
        )

    def _get_data(self, flag: str):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _ensure_visual_dirs(self, setting: str):
        root = Path(self.args.visual_root) / setting
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "videos").mkdir(parents=True, exist_ok=True)
        return root

    def _maybe_export_batch_visuals(
        self, setting: str, batch_idx: int, batch_x: torch.Tensor, var_names: list[str] | None
    ):
        """
        每个 batch 单独一个文件夹:
        - images/batch_{i}/ 里放 32 张图片 (batch_size=32)
        - videos/batch_{i}/ 里放 32 个视频 (batch_size=32)
        """
        root = self._ensure_visual_dirs(setting)

        if self.args.gen_images_batches > 0 and batch_idx < self.args.gen_images_batches:
            out_dir = root / "images" / f"batch_{batch_idx:04d}"
            try:
                export_batch_correlation_images(
                    batch_x=batch_x.detach().cpu(),
                    out_dir=str(out_dir),
                    max_samples=self.args.batch_size,
                    corr_n_vars=self.args.corr_n_vars,
                    var_names=var_names,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[warn] image export failed for batch={batch_idx}: {e}")

        if self.args.gen_videos_batches > 0 and batch_idx < self.args.gen_videos_batches:
            out_dir = root / "videos" / f"batch_{batch_idx:04d}"
            try:
                export_batch_patch_correlation_videos(
                    batch_x=batch_x.detach().cpu(),
                    out_dir=str(out_dir),
                    patch_size=self.args.patch_size,
                    stride=self.args.patch_stride,
                    fps=self.args.video_fps,
                    max_samples=self.args.batch_size,
                    corr_n_vars=self.args.corr_n_vars,
                    var_names=var_names,
                    ffmpeg_path=self.args.ffmpeg_path,
                )
            except Exception as e:  # noqa: BLE001
                # 如果你明确开启了视频导出，这里直接报错更好（避免默默生成不了 mp4）
                raise RuntimeError(f"video export failed for batch={batch_idx}: {e}") from e

    def train(self, setting: str):
        train_data, train_loader = self._get_data(flag="train")
        val_data, val_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        var_names = getattr(train_data, "feature_names", None)
        if var_names is not None and self.args.corr_n_vars and self.args.corr_n_vars > 0:
            var_names = list(var_names)[: self.args.corr_n_vars]

        # 打印切分后的数据形状（Informer 风格的 DataLoader 输出）
        # X = batch_x: encoder 输入窗口 [B, seq_len, n_features]
        # Y = batch_y: decoder 输入+预测窗口 [B, label_len+pred_len, n_features]
        batch_x0, batch_y0, batch_x_mark0, batch_y_mark0 = next(iter(train_loader))
        # print(f"[shape] 第一批数据形状：X={tuple(batch_x0.shape)} Y={tuple(batch_y0.shape)} X_mark={tuple(batch_x_mark0.shape)} Y_mark={tuple(batch_y_mark0.shape)}")

        x0 = batch_x0
        if self.args.corr_n_vars and self.args.corr_n_vars > 0:
            x0 = x0[:, :, : self.args.corr_n_vars]
        bsz, seq_len, n_vars = x0.shape
        if var_names is None:
            var_names_out = [f"v{i}" for i in range(n_vars)]
        else:
            var_names_out = list(var_names)[:n_vars]

        mean_bt = x0.mean(dim=1).detach().cpu()  # [B, C] over T
        var_bt = x0.var(dim=1, unbiased=False).detach().cpu()  # [B, C] over T

        # print(f"[stat] 第一批数据(标准化后)：样本数={bsz} 时间点数T={seq_len} 变量数={n_vars}")
        # for sample_idx in range(bsz):
        #     for var_idx in range(n_vars):
        #         vname = var_names_out[var_idx]
        #         m = float(mean_bt[sample_idx, var_idx].item())
        #         v = float(var_bt[sample_idx, var_idx].item())
        #         print(f"[stat] 第{sample_idx+1}个样本的第{var_idx+1}个变量是{vname}，均值={m:.6f}，方差={v:.6f}")
        patches0 = x0.unfold(dimension=1, size=self.args.patch_size, step=self.args.patch_stride)
        print(f"[shape] train batch_x patches = {tuple(patches0.shape)}  # [B, n_patches, patch_len, n_vars]")

        criterion = nn.MSELoss()
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.model.train()
        for epoch in range(self.args.train_epochs):
            total_loss = 0.0
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 可视化导出（默认只导出第一批，改大很慢）
                self._maybe_export_batch_visuals(
                    setting=setting,
                    batch_idx=batch_idx,
                    batch_x=batch_x,
                    var_names=var_names,
                )

                model_optim.zero_grad()
                pred = self.model(batch_x, batch_x_mark, None, None)

                # 训练目标: 只取未来 pred_len 段做回归
                true = batch_y[:, -self.args.pred_len :, :]
                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()

                total_loss += loss.item()

                if (batch_idx + 1) % self.args.log_step == 0:
                    print(f"epoch={epoch+1}/{self.args.train_epochs} step={batch_idx+1}/{len(train_loader)} loss={loss.item():.6f}")

            print(f"epoch={epoch+1} train_loss={total_loss/len(train_loader):.6f}")

        # quick evaluation on test
        self.test(setting=setting)

        return self.model

    @torch.no_grad()
    def test(self, setting: str):
        _, test_loader = self._get_data(flag="test")
        criterion = nn.MSELoss()
        self.model.eval()

        total_loss = 0.0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred = self.model(batch_x, batch_x_mark, None, None)
            true = batch_y[:, -self.args.pred_len :, :]
            total_loss += criterion(pred, true).item()

        print(f"test_loss={total_loss/len(test_loader):.6f}")
