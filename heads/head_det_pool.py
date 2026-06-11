import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_base_det import HeadBaseDet

class HeadDetPool(HeadBaseDet):
    def __init__(self,
                 *args,
                 n_views=1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.n_views = n_views
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        x, mask = inputs
        if len(x.shape) > 3 and self.n_views > 1:
            B, P, T, C = x.shape
            assert P % self.n_views == 0, f"{P} vs {self.n_views}"
            x = x.reshape(B, self.n_views, P//self.n_views, T, C)
            x = x.permute(0, 1, 3, 4, 2).flatten(0, 3)
            pool = nn.AdaptiveAvgPool1d(1)
            x = pool(x)
            x = x.reshape(B, self.n_views, T, C)
            x = x.mean(1)

        # [N, C, W, 1, P]
        if len(x.shape) == 5 and x.shape[3] == 1:# and x.shape[4] == 1:
            x = x.squeeze(-2)#.squeeze(-1) # [N, C, W, P]
            B, C, T, P = x.shape
            # pool on persons
            if x.shape[-1] > 1:
                x = x.reshape(B, C*T, P)
                pool = nn.AdaptiveAvgPool1d(1)
                x = pool(x)
            x = x.reshape(B, C, T) # [N, C, W]
            # x = x.squeeze(-1) # 
            x = x.permute(0, 2, 1) # [N, W, C]
            x = self.fc(x)

        return x, mask
