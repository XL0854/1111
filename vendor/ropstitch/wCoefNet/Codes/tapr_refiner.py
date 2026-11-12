import torch
import torch.nn as nn
import torch.nn.functional as F
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU()
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class TAPRRefiner(nn.Module):
    """
    Topology-Aware Residual Refiner (TAPR):
    Takes aligned features, warped overlap image difference, and predicts
    residual vertex displacements (delta_motion) on the (grid_h+1, grid_w+1) TPS mesh.
    
    Zero-initialization ensures:
    At step 0: delta_motion == 0, output mesh == baseline mesh exactly.
    """
    def __init__(self, feat_dim=256, diff_in_c=3, hidden_dim=64):
        super(TAPRRefiner, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Downsample 512x512 diff image to 64x64 (matching feature map size)
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(diff_in_c, 16, kernel_size=3, stride=2, padding=1),  # 256
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # 128
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),          # 64
            nn.GELU(),
        )
        
        total_in_c = feat_dim + 32  # 256 (feat) + 32 (diff) = 288
        
        self.in_proj = nn.Sequential(
            nn.Conv2d(total_in_c, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU()
        )
        
        # Multi-scale contextual residual blocks
        self.res1 = ConvBlock(hidden_dim, hidden_dim)
        self.res2 = ConvBlock(hidden_dim, hidden_dim)
        self.res3 = ConvBlock(hidden_dim, hidden_dim)
        
        # Map 64x64 features to target control point grid resolution (grid_h+1, grid_w+1)
        self.pool_to_grid = nn.AdaptiveAvgPool2d((self.grid_h + 1, self.grid_w + 1))
        
        # Vertex residual heads for ref and tgt meshes
        self.head_ref = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1)
        )
        self.head_tgt = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1)
        )
        
        # Maximum displacement scale (limits refiner to realistic local adjustments, e.g. max 15 pixels)
        self.max_disp = 15.0
        
        # CRITICAL: Zero-initialization
        self._init_zeros()

    def _init_zeros(self):
        # Set weights and bias of final convs to zero
        nn.init.zeros_(self.head_ref[-1].weight)
        nn.init.zeros_(self.head_ref[-1].bias)
        nn.init.zeros_(self.head_tgt[-1].weight)
        nn.init.zeros_(self.head_tgt[-1].bias)

    def forward(self, warp_f1, warp_f2, diff_img):
        """
        warp_f1: [B, 64, H/8, W/8] (e.g. [B, 64, 64, 64])
        warp_f2: [B, 64, H/8, W/8]
        diff_img: [B, 3, H, W] (e.g. [B, 3, 512, 512])
        Returns:
            delta_ref: [B, grid_h+1, grid_w+1, 2]
            delta_tgt: [B, grid_h+1, grid_w+1, 2]
        """
        diff_feat = self.diff_encoder(diff_img)  # [B, 32, 64, 64]
        feat_cat = torch.cat([warp_f1, warp_f2, diff_feat], dim=1)  # [B, 160, 64, 64]
        
        h = self.in_proj(feat_cat)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        
        grid_feat = self.pool_to_grid(h)  # [B, hidden_dim, 13, 13]
        
        # Predict delta with tanh scaling (bounded within [-max_disp, max_disp])
        delta_ref = torch.tanh(self.head_ref(grid_feat)) * self.max_disp  # [B, 2, 13, 13]
        delta_tgt = torch.tanh(self.head_tgt(grid_feat)) * self.max_disp  # [B, 2, 13, 13]
        
        # Transpose to [B, 13, 13, 2] to match mesh format
        delta_ref = delta_ref.permute(0, 2, 3, 1)
        delta_tgt = delta_tgt.permute(0, 2, 3, 1)
        
        return delta_ref, delta_tgt
