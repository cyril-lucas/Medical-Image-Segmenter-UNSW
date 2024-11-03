import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock

# Custom Layer: Patch Embedding
class PatchEmbedding(nn.Module):
    """Embedding layer that extracts patches and projects them into a specified embedding dimension."""
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [B, embed_dim, H//patch_size, W//patch_size]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Reshape to [B, N, embed_dim] where N is the number of patches
        return x, H, W

# Custom Layer: Patch Merging
class PatchMerging(nn.Module):
    """Merges patches for downsampling."""
    def __init__(self, input_dim):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input features has wrong size"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # Shape: [B, H//2, W//2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # Concatenate along the channel dimension

        x = x.view(B, -1, 4 * C)  # Reshape to [B, (H//2)*(W//2), 4*C]
        x = self.reduction(x)  # Linear reduction to [B, (H//2)*(W//2), 2*C]
        return x

# Custom Layer: Patch Expanding
class PatchExpanding(nn.Module):
    """Expands patches for upsampling."""
    def __init__(self, input_dim, upsample_rate=2):
        super(PatchExpanding, self).__init__()
        self.upsample_rate = upsample_rate
        self.linear = nn.Linear(input_dim, input_dim // 2)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input features has wrong size"

        x = self.linear(x)  # Shape: [B, L, C//2]
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # Reshape to [B, C//2, H, W]
        x = F.interpolate(x, scale_factor=self.upsample_rate, mode='bilinear', align_corners=False)  # Upsample
        H, W = H * self.upsample_rate, W * self.upsample_rate  # Update height and width
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose back to [B, H*W, C//2]
        return x, H, W

# Swin U-Net Model
class SwinUNet2DBase(nn.Module):
    def __init__(self, in_channels, filter_num_begin, depth, stack_num_down, stack_num_up,
                 patch_size, num_heads, window_size, num_mlp, shift_window=True):
        super(SwinUNet2DBase, self).__init__()
        self.embed_dim = filter_num_begin
        self.depth = depth
        self.shift_window = shift_window

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels, self.embed_dim, patch_size)

        # Encoder: Swin Transformer + Patch Merging
        self.swin_down = nn.ModuleList([
            SwinTransformerBlock(dim=self.embed_dim * (2 ** i),
                                 num_heads=num_heads[i],
                                 window_size=window_size[i],
                                 shift_size=window_size[i] // 2 if shift_window else 0,
                                 mlp_ratio=num_mlp)
            for i in range(depth)
        ])
        self.patch_merging = nn.ModuleList([
            PatchMerging(self.embed_dim * (2 ** i))
            for i in range(depth - 1)
        ])

        # Decoder: Patch Expanding + Swin Transformer
        self.swin_up = nn.ModuleList([
            SwinTransformerBlock(embed_dim=self.embed_dim * (2 ** (depth - i - 1)),
                                 num_heads=num_heads[depth - i - 1],
                                 window_size=window_size[depth - i - 1],
                                 shift_size=window_size[depth - i - 1] // 2 if shift_window else 0,
                                 mlp_ratio=num_mlp)
            for i in range(depth)
        ])
        self.patch_expanding = nn.ModuleList([
            PatchExpanding(self.embed_dim * (2 ** (depth - i - 1)))
            for i in range(depth - 1)
        ])

    def forward(self, x):
        # Initial Patch Embedding
        x, H, W = self.patch_embedding(x)

        # Encoder
        skip_connections = []
        for i in range(self.depth):
            x = self.swin_down[i](x)
            skip_connections.append(x)
            if i < self.depth - 1:
                x = self.patch_merging[i](x, H, W)
                H, W = H // 2, W // 2  # Update height and width

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            if i > 0:
                x, H, W = self.patch_expanding[i - 1](x, H, W)
            x = torch.cat([x, skip_connections[i]], dim=-1)  # Concatenate skip connections
            x = self.swin_up[i](x)

        return x

# Example usage
model = SwinUNet2DBase(in_channels=3, filter_num_begin=24, depth=4,
                       stack_num_down=2, stack_num_up=2,
                       patch_size=4, num_heads=[3, 6, 12, 24],
                       window_size=[7, 7, 7, 7], num_mlp=4)
dummy_input = torch.randn(1, 3, 256, 256)
output = model(dummy_input)

print(output.shape)  # Expected shape: [1, H*W, C]
