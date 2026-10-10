import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherFusionScorer(nn.Module):
    """
    Teacher network: a frozen-parameter dual-branch fusion scorer
    Directly uses the pretrained weights of the existing ResNetAutoEncoder3D-18 and ResNet50 encoder_q
    """
    def __init__(self, model_ae, model_moco, lambda_reg=0.2, lambda_dis=0.8):
        super().__init__()
        self.model_ae = model_ae
        self.model_moco = model_moco
        self.lambda_reg = lambda_reg
        self.lambda_dis = lambda_dis

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, template_reg, template_dis, candidate_reg, candidate_dis):
        """
        Args:
            template_reg:   [B, 1, D, H, W]   (template, AE branch, typically 32^3)
            template_dis:   [B, 1, D, H, W]   (template, MoCo branch, typically 64^3)
            candidate_reg:  [B*N, 1, D, H, W] (candidate, AE branch)
            candidate_dis:  [B*N, 1, D, H, W] (candidate, MoCo branch)
        Returns:
            scores:     [B*N]  fused similarity score
            ae_sim:     [B*N]  AE branch cosine similarity
            moco_sim:   [B*N]  MoCo branch NCC similarity
            x1:         [B*N, 256]  candidate AE features (used for feature distillation)
        """
        # AE branch features
        z1 = self.model_ae.features(template_reg, mlp=True)   # [B, 256]
        x1 = self.model_ae.features(candidate_reg, mlp=True)  # [B*N, 256]

        # Expand the template features to match the number of candidates
        B = z1.shape[0]
        N_total = x1.shape[0]
        N = N_total // B
        z1 = z1.repeat_interleave(N, dim=0)  # [B*N, 256]
        ae_sim = F.cosine_similarity(z1, x1, dim=1)  # [B*N]

        # MoCo branch features
        z2 = self.model_moco(template_dis)                    # [B, 2048, D, H, W]
        x2 = self.model_moco(candidate_dis)                   # [B*N, 2048, D, H, W]
        z2 = z2.repeat_interleave(N, dim=0)  # [B*N, 2048, D, H, W]
        moco_sim = self.calculate_ncc(z2, x2)                 # [B*N]

        scores = self.lambda_reg * ae_sim + self.lambda_dis * moco_sim  # [B*N]
        return scores, ae_sim, moco_sim, x1

    @staticmethod
    def calculate_ncc(z, x, eps=1e-8):
        """
        Compute normalized cross-correlation (NCC)
        Kept consistent with MultiBranchTracker.calculate_ncc
        """
        # Automatically add batch and channel dimensions (if missing)
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            z = z.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(0)  # (1, C, D, H, W)
            z = z.unsqueeze(0)

        B, C, D, H, W = z.shape
        z_flat = z.view(B, C, -1)
        x_flat = x.view(B, C, -1)

        # Compute the mean (along the spatial dimensions)
        x_mean = x_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
        z_mean = z_flat.mean(dim=2, keepdim=True)

        # Mean subtraction
        x_centered = x_flat - x_mean
        z_centered = z_flat - z_mean

        # Numerator: covariance term
        cov = (x_centered * z_centered).sum(dim=2)  # (B, C)

        # Denominator: product of the respective standard deviations
        x_var = (x_centered ** 2).sum(dim=2)
        z_var = (z_centered ** 2).sum(dim=2)
        std_prod = torch.sqrt(x_var * z_var + eps)  # (B, C)

        # NCC per channel
        ncc_per_channel = cov / std_prod  # (B, C)

        # Average over channels
        ncc = ncc_per_channel.mean(dim=1)  # (B,)

        return ncc
