import copy as cp
import torch
import torch.nn as nn
from .utils import Graph

EPS = 1e-4


# ======================================================
# GCN BLOCK
# ======================================================
class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, residual=True, **kwargs):
        super().__init__()

        self.gcn = unit_gcn(in_channels, out_channels, A)
        self.tcn = mstcn(out_channels, out_channels)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x, _ = self.gcn(x)
        x = self.tcn(x) + res

        return self.relu(x)


# ======================================================
# ProtoGCN
# ======================================================
class SWGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10):

        super().__init__()

        # Graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        # Data normalization (stable for short T)
        self.data_bn = nn.LayerNorm(in_channels * A.size(1))

        modules = []
        if in_channels != base_channels:
            modules.append(
                GCN_Block(in_channels, base_channels, A, residual=False)
            )

        inflate_times = 0
        channels = base_channels

        for i in range(2, num_stages + 1):
            in_c = channels
            if i in [5, 8]:
                inflate_times += 1
            out_c = int(base_channels * (ch_ratio ** inflate_times) + EPS)
            channels = out_c

            modules.append(
                GCN_Block(in_c, out_c, A)
            )

        self.gcn = nn.ModuleList(modules)
        self.out_channels = channels

    # --------------------------------------------------
    def forward(self, inputs):
        """
        x: [N, M, T, V, C]
        """
        if isinstance(inputs, tuple):
            x, mask = inputs
        else:
            x  = inputs

        if len(x.size()) == 6:
            N, W, M, T, V, C = x.size()
            x = x.view(N * W, M, T, V, C)
            N = N * W
        else:
            W = 1
            N, M, T, V, C = x.size()

        # -------- Data BN --------
        x = x.permute(0, 1, 3, 4, 2).contiguous()      # N M V C T
        x = x.view(N * M, V * C, T).transpose(1, 2)   # N*M T VC
        x = self.data_bn(x)
        x = x.transpose(1, 2).view(N * M, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous().view(N * M, C, T, V)

        # -------- GCN Backbone --------
        for block in self.gcn:
            x = block(x)

        # -------- Pooling for TAD --------
        x = x.view(N, M, self.out_channels, T, V)
        # x = x.mean(dim=1)    # persons
        # x = x.mean(dim=-1)   # joints
        # x = x.mean(dim=-1)   # time

        if W > 1:

            # Global Average Pooling
            pool = nn.AdaptiveAvgPool3d(1)
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, self.out_channels, M, T, V)
            x = pool(x)
            x = x.squeeze(-1)
            x = x.squeeze(-1)
            x = x.squeeze(-1)
            # x: [N, C]

            x = x.view(N // W, W, self.out_channels) # N, W, C

            x = x.permute(0, 2, 1).contiguous()  # N, C, W
            x = x.unsqueeze(-1).unsqueeze(-1)  # N, C, W, 1, 1

        if isinstance(inputs, tuple):
            x = (x, mask)
        return x


# ======================================================
# UNIT GCN
# ======================================================
class unit_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, ratio=0.125):
        super().__init__()

        self.A = nn.Parameter(A.clone())
        self.num_subsets = A.size(0)

        mid_channels = int(out_channels * ratio)

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1),
            nn.GroupNorm(32, mid_channels * self.num_subsets),
            nn.ReLU()
        )

        self.post = nn.Conv2d(mid_channels * self.num_subsets, out_channels, 1)
        self.bn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels)
            )
        else:
            self.down = lambda x: x

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.down(x)

        A = self.A[None, :, None, :, :]  # 1 K 1 V V

        x = self.pre(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum('nkctv,nkcvw->nkctw', x, A)
        x = x.reshape(n, -1, t, v)

        x = self.post(x)
        x = self.bn(x) + res
        return self.relu(x), None


# ======================================================
# MSTCN (TRF contrôlé)
# ======================================================
class mstcn(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branches = nn.ModuleList([
            unit_tcn(in_channels, out_channels // 2, kernel_size=3, dilation=1),
            unit_tcn(in_channels, out_channels // 2, kernel_size=3, dilation=2),
        ])

        self.transform = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        return self.transform(x)


# ======================================================
# UNIT TCN
# ======================================================
class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        pad = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.GroupNorm(16, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))