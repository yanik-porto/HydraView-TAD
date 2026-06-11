import torch
import torch.nn as nn
import torch.nn.functional as F

# from encoder.encoders.temporal_encoders import create_temporal_encoder

class HeadBaseDet(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_persons=1,
                 n_views=1,
                 loss='bce',
                 temporal_encoder=None,
                 **kwargs):
        self.num_classes = num_classes
        self.in_c = in_channels
        self.n_persons = n_persons
        self.n_views = n_views
        super().__init__(**kwargs)

        self.ce = nn.CrossEntropyLoss()
        if loss == 'bce':
            self.criterion = BCEWithMask()
        elif loss == 'ce':
            self.criterion = ContrastiveInfoNCELoss()
        else:
            raise ValueError(f'Unknown loss {loss}')

        # self.temporal_encoder = None
        # if temporal_encoder is not None:
        #     te_name = temporal_encoder.get("name", None)
        #     te_params = temporal_encoder.get("params", {})
        #     self.temporal_encoder = create_temporal_encoder(te_name, **te_params)

        self.fc = nn.Linear(self.in_c, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def forward(self, inputs):

        # if self.temporal_encoder is not None:
        #     inputs = self.temporal_encoder(inputs)

        x, mask = inputs

        x = self.fc(x)

        # pool on persons and views
        if len(x.shape) > 3:
            B, P, T, C = x.shape
            pool = nn.AdaptiveAvgPool1d(1)
            x = x.permute(0, 2, 3, 1).flatten(0, 2)
            x = pool(x)
            x = x.reshape(B, T, C)

        return x, mask

    def loss(self, output, labels):
        return self.ce(output, labels)

    def loss_bce(self, output, labels):
        loss = self.criterion(output, labels)
        return loss

class BCEWithMask(nn.Module):
    def __init__(self):
        super(BCEWithMask, self).__init__()

    def forward(self, output, labels):
        outputs_final, mask = output
        if outputs_final.dim() > 3:
            outputs_final = outputs_final[0,:,:,:]
        if outputs_final.shape[1] != labels.shape[1]:
            outputs_final = outputs_final.permute(0, 2, 1) # TODO : maybe check how to avoid this permute for mstemba

        loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
        loss_f = torch.sum(loss_f) / torch.sum(mask)
        return loss_f

# Define the contrastive loss
class ContrastiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveInfoNCELoss, self).__init__()

        self.temperature = temperature

    def forward(self, z):
        z, mask = z
        # Extract embeddings
        z_anchor = z[:, 0]  # (B, C)
        z_pos    = z[:, 1]  # (B, C)
        z_neg    = z[:, 2]  # (B, C)

        # Normalize
        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos    = F.normalize(z_pos, dim=1)
        z_neg    = F.normalize(z_neg, dim=1)

        # Cosine similarities
        sim_pos = torch.sum(z_anchor * z_pos, dim=1) / self.temperature  # (B,)
        sim_neg = torch.sum(z_anchor * z_neg, dim=1) / self.temperature  # (B,)

        # InfoNCE loss
        loss = -torch.log(
            torch.exp(sim_pos) /
            (torch.exp(sim_pos) + torch.exp(sim_neg))
        )

        return loss.mean()