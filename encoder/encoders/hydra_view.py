import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmamba import VSSBlock
from .vision_mamba import VisionMamba

class TemporalConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=dilation*(kernel_size//2), dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (B, T, M, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, T, M)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, M, C)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class HydraView(nn.Module):
    def __init__(self, *args,
                    # num_classes,
                    in_feat_dim=256,
                    embed_dims=[256, 384, 576, 864],
                    depths=[1, 1, 1, 1],
                    d_state=16,
                    n_persons=1,
                    do_align=False,
                    **kwargs
                 ):
        super(HydraView, self).__init__()

        self.n_persons = n_persons

        self.blocks = nn.ModuleList()
        # self.patch_embed = self.make_patch_embed(in_feat_dim, embed_dims[0], patch_size=(1, 2))
        drop_path_rate = 0.3
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        for i in range(len(depths)):

            if i == 0:
                self.blocks.append(TemporalConv2DBlock(embed_dims[0], embed_dims[i], kernel_size=3, stride=(1, 2), dilation=i+1))
            else:
                self.blocks.append(TemporalConv2DBlock(embed_dims[i-1], embed_dims[i], kernel_size=3, stride=(1, 2), dilation=i+1))
            
            drop_path = dpr[sum(depths[:i]):sum(depths[:i + 1])],
            self.blocks.append(self._create_vmamba_block(embed_dims[i], d_state, depths[i], drop_path))
    
        self.proj = nn.Linear(in_feat_dim, embed_dims[0])

        # Fusion projections
        self.scale_proj1 = nn.Linear(embed_dims[0], embed_dims[2])
        self.scale_proj2 = nn.Linear(embed_dims[1], embed_dims[2])
        self.scale_proj3 = nn.Linear(embed_dims[2], embed_dims[2])

        # self.interaction_block = self._create_vmamba_block(embed_dims[-1], d_state, depths[i], drop_path)
        self.interaction_block = self._create_vision_mamba_block(embed_dims[-1], d_state, depths[i])
        self.norm = nn.LayerNorm(embed_dims[-1])


        # self.head = self.make_classifier(embed_dims, 5)

    def _create_vision_mamba_block(self, embed_dim, d_state, depth):
        return VisionMamba(
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            final_pool_type='all',
            if_abs_pos_embed=False,
            if_rope=False, 
            if_rope_residual=False, 
            bimamba_type="v2", 
            if_cls_token=False, 
            if_divide_out=True, 
            use_middle_cls_token=True,
        )

    def _create_vmamba_block(self, embed_dim, d_state, depth, drop_path):
            depth = len(drop_path)

            blocks = []
            for d in range(depth):
                blocks.append(VSSBlock(
                    hidden_dim=embed_dim, 
                    # drop_path=drop_path[d],
                    channel_first=False,
                    ssm_d_state=d_state,
                    ssm_ratio=2.0,
                    ssm_dt_rank="auto",
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
                    ssm_init="v0", forward_type="v0", 
                    mlp_ratio=0.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                    use_checkpoint=False,
                ))
            return nn.Sequential(*blocks,)
    

    def make_classifier(self, embed_dims, num_classes):
        return nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def fuse_views(self, x):
        N, T, M, C = x.shape
        x = x.permute(0, 1, 3, 2)  # (B, T, C, M)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        pool = nn.AdaptiveAvgPool1d(1)
        x = pool(x).squeeze(-1)
        x = x.reshape(N, T, -1)
        return x

    def forward_features(self, x):

        x = self.proj(x)

        concat_x = []
        for i, block in enumerate(self.blocks):
            if i == 0 or i == 1:  # First block - no reshaping
                if isinstance(block, TemporalConv2DBlock):
                    x = block(x)
                else:
                    x = block(x)
                    concat_x.append(self.fuse_views(x))
            elif i == 2 or i == 3:  # Second block - reshape to 2B
                if isinstance(block, TemporalConv2DBlock):
                    x = block(x)
                    # Reshape to (2B, T/2, C)
                    B, T, M, C = x.shape
                    x = x.reshape(B, 2, T//2, M, C).transpose(0, 1).reshape(2*B, T//2, M, C)
                else:  # VMamba
                    x = block(x)
                    # Reshape back from (2B, T/2, C) to (B, T, C)
                    B2, T_half, M, C = x.shape
                    B = B2 // 2
                    x = x.reshape(2, B, T_half, M, C).transpose(0, 1).reshape(B, T_half*2, M, C)
                    concat_x.append(self.fuse_views(x))
            elif i == 4 or i == 5:  # Third block - reshape to 3B
                if isinstance(block, TemporalConv2DBlock):
                    x = block(x)    
                    # Reshape to (3B, T/3, C)
                    B, T, M, C = x.shape
                    # Ensure T is divisible by 3
                    pad_size = (3 - (T % 3)) % 3  # Calculate padding needed
                    if pad_size > 0:
                        x = F.pad(x, (0, 0, 0, 0, 0, pad_size))  # Pad along temporal dimension
                        T = T + pad_size
                    x = x.reshape(B, 3, T//3, M, C).transpose(0, 1).reshape(3*B, T//3, M, C)

                else:  # VMamba 
                    x = block(x)
                    # Reshape back from (3B, T/3, C) to (B, T, C)
                    B3, T_third, M, C = x.shape
                    B = B3 // 3
                    x = x.reshape(3, B, T_third, M, C).transpose(0, 1).reshape(B, T_third*3, M, C)
                    # Remove padding if it was added
                    if pad_size > 0:
                        x = x[:, :-pad_size, :]
                    concat_x.append(self.fuse_views(x))

        return concat_x

    def forward(self, x):
        x, mask = x # (N, C, T, M)

        # stack
        x = x.squeeze(3)
        assert len(x.shape) == 4, x.shape
        N, C, T, M = x.shape

        if self.n_persons > 1:
            assert M % self.n_persons == 0, f"{M} vs {self.n_persons}"
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(N*self.n_persons, M//self.n_persons, C, T)
            N = N*self.n_persons
            M = M//self.n_persons
            x = x.permute(0, 2, 3, 1)

        x = x.permute(0, 2, 3, 1) # (N, T, M, C)

        x = self.forward_features(x)

        # Fusion and Mamba interaction
        x1, x2, x3 = x

        x1 = self.norm(self.scale_proj1(x1))
        x2 = self.norm(self.scale_proj2(x2))
        x3 = self.norm(self.scale_proj3(x3))

        x = x1 + x2 + x3

        x = self.interaction_block(x)

        # x = x.permute(0, 1, 3, 2)  # (B, T, C, M)
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        # pool = nn.AdaptiveAvgPool1d(1)
        # x = pool(x).squeeze(-1)
        # x = x.reshape(N, T, -1)

        # pool on persons
        if self.n_persons > 1:
            pool = nn.AdaptiveAvgPool1d(1)
            N = N//self.n_persons
            x = x.reshape((N, self.n_persons) + x.shape[1:])
            x = x.permute(0, 2, 3, 1)
            x = x.flatten(0,2)
            x = pool(x)
            x = x.reshape(N, T, -1)

        # x = self.head(x)

        return x, mask
