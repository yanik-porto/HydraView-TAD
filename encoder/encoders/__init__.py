from .unik import *
from .protogcn import *
from .swgcn import *
from .pdan import PDAN
from .ms_temba import MSTemba
from .hydra_view import HydraView

def create_encoder(config, nodename='encoder'):
    assert nodename in config, "no encoder specified in config"

    cfg_encoder = config[nodename]

    name = cfg_encoder.get('name', None)

    encoder = None
    if name == "unik":
        encoder = UNIK(**cfg_encoder["params"])
    elif name == "protogcn":
        encoder = ProtoGCN(**cfg_encoder["params"])
    elif name == "swgcn":
        encoder = SWGCN(**cfg_encoder["params"])
    elif name == "pdan":
        encoder = PDAN(**cfg_encoder["params"])
    elif name == "ms_temba":     
        encoder = MSTemba(
            embed_dims=[256, 384, 576],
            temporal_dims=[256],
            depths=[1, 1, 1],
            d_state=16,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            **cfg_encoder["params"])
    elif name == "hydra_view":
        encoder = HydraView(
            temporal_dims=[384],
            d_state=16,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            **cfg_encoder["params"])
    elif name == "empty":
        class EmptyEncoder(nn.Module):
            def forward(self, x):
                return x
        encoder = EmptyEncoder()
    else:
        print(name, " not handled yet in encoders")

    return encoder