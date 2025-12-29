import torch


def patchify_ts(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: [B, L, C] -> tokens: [B, (num_patches*C), patch_len]
    """
    B, L, C = x.shape
    num_patches = (L - patch_len) // stride + 1
    if (L - patch_len) % stride != 0:
        print(f"[WARN] L={L}, patch_len={patch_len}, stride={stride} not aligned; last part truncated by formula.")
    patches = []
    for p in range(num_patches):
        start = p * stride
        patches.append(x[:, start:start + patch_len, :])  # [B, patch_len, C]
    patches = torch.stack(patches, dim=0).permute(1, 0, 2, 3).contiguous()  # [B, num_patches, patch_len, C]
    patches = patches.permute(0, 3, 1, 2).contiguous()  # [B, C, num_patches, patch_len]
    tokens = patches.view(B, C * num_patches, patch_len).contiguous()
    return tokens
