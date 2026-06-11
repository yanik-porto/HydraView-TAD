# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_mamba import VisionMamba

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=dilation*(kernel_size//2), dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.activation(x)
        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class MSTemba(nn.Module):
    def __init__(self, 
                 in_feat_dim=768, #CLIP
                #  in_feat_dim=1024, #I3D
                #  in_chans=3, 
                #  num_classes=157,
                 embed_dims=[256, 384, 576, 864],
                 temporal_dims=[256, 128, 64, 32],
                 depths=[1, 1, 1, 1],
                 d_state=16,
                 **kwargs):
        super().__init__()
        
        # self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.temporal_dims = temporal_dims
        tsu_temporal_dim = 2500
        ch_temporal_dim = 256
        mth_temporal_dim = 2000
        self.proj = nn.Linear(in_feat_dim, embed_dims[0])

        self.scale_proj1 = nn.Linear(embed_dims[0], embed_dims[2])
        self.scale_proj2 = nn.Linear(embed_dims[1], embed_dims[2])
        self.scale_proj3 = nn.Linear(embed_dims[2], embed_dims[2])

        # Hierarchical blocks
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            # Temporal convolution
            if i == 0:
                self.blocks.append(TemporalConvBlock(embed_dims[0], embed_dims[i], kernel_size=3, stride=1, dilation=i+1))
            else:
                self.blocks.append(TemporalConvBlock(embed_dims[i-1], embed_dims[i], kernel_size=3, stride=1, dilation=i+1))
            
            # Vision Mamba block
            self.blocks.append(self._create_mamba_block(embed_dims[i], d_state, depths[i], **kwargs))


        self.interaction_block = self._create_mamba_block(embed_dims[-1], d_state, depths[i], **kwargs)

        # Final norm and classifier
        self.norm = nn.LayerNorm(embed_dims[-1])
        # self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _create_mamba_block(self, embed_dim, d_state, depth, **kwargs):
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        concat_x = []
        for i, block in enumerate(self.blocks):
            if i == 0 or i == 1:  # First block - no reshaping
                if isinstance(block, TemporalConvBlock):
                    x = block(x)
                else:  # VisionMamba 
                    B, T, C = x.shape
                    x = block.forward_features(x)
                    concat_x.append(x)
            
            # else:  # Second block - reshape to 2B
            elif i == 2 or i == 3:  # Second block - reshape to 2B
                if isinstance(block, TemporalConvBlock):
                    x = block(x)    
                    # Reshape to (2B, T/2, C)
                    B, T, C = x.shape
                    x = x.reshape(B, 2, T//2, C).transpose(0, 1).reshape(2*B, T//2, C)
                else:  # VisionMamba 
                    x = block.forward_features(x)
                    # Reshape back from (2B, T/2, C) to (B, T, C)
                    B2, T_half, C = x.shape
                    B = B2 // 2
                    x = x.reshape(2, B, T_half, C).transpose(0, 1).reshape(B, T_half*2, C)
                    concat_x.append(x)

            elif i == 4 or i == 5:  # Third block - reshape to 3B
                if isinstance(block, TemporalConvBlock):
                    x = block(x)    
                    # Reshape to (3B, T/3, C)
                    B, T, C = x.shape
                    # Ensure T is divisible by 3
                    pad_size = (3 - (T % 3)) % 3  # Calculate padding needed
                    if pad_size > 0:
                        x = F.pad(x, (0, 0, 0, pad_size))  # Pad along temporal dimension
                        T = T + pad_size
                    x = x.reshape(B, 3, T//3, C).transpose(0, 1).reshape(3*B, T//3, C)

                else:  # VisionMamba 
                    x = block.forward_features(x)
                    # Reshape back from (3B, T/3, C) to (B, T, C)
                    B3, T_third, C = x.shape
                    B = B3 // 3
                    x = x.reshape(3, B, T_third, C).transpose(0, 1).reshape(B, T_third*3, C)
                    # Remove padding if it was added
                    if pad_size > 0:
                        x = x[:, :-pad_size, :]
                    concat_x.append(x)

        return concat_x

    def forward(self, x):

        inputs, mask = x
        
        B, C, T, _, P = inputs.shape

        # pool on persons if needed
        if P != 1:
            # Pool
            # pool_person = nn.AdaptiveAvgPool1d(1)
            # inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
            # inputs = pool_person(inputs)
            # inputs = inputs.reshape(B, C, T, 1, 1)

            # Permute
            inputs = inputs.permute(0, 4, 1, 2, 3).contiguous().view(B, P, C, T, 1)
            inputs = inputs.reshape(B*P, C, T, 1, 1)

        

        inputs = inputs.squeeze(3).squeeze(3)
        
        x = self.forward_features(inputs)

        # Fusion and Mamba interaction
        x1, x2, x3 = x

        x1 = self.norm(self.scale_proj1(x1))
        x2 = self.norm(self.scale_proj2(x2))
        x3 = self.norm(self.scale_proj3(x3))
    
        x = x1 + x2 + x3

        x = self.interaction_block(x)

        # x = self.head(x)

        if P != 1:
            x = x.reshape((B, P) + x.shape[1:])

        return x, mask
