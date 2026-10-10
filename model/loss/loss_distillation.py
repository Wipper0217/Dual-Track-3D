import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Three-level distillation loss: response distillation + feature distillation + ranking distillation
    """
    def __init__(self, w_response=1.0, w_feature=0.5, w_rank=0.3, margin=0.3):
        super().__init__()
        self.w_response = w_response
        self.w_feature = w_feature
        self.w_rank = w_rank
        self.margin = margin

    def forward(self, student_feat, student_score,
                teacher_feat_ae, teacher_score,
                pos_mask=None, neg_mask=None):
        """
        Args:
            student_feat:    [N, 256]  student network candidate features
            student_score:   [N]       student network similarity score (cosine_sim)
            teacher_feat_ae: [N, 256]  teacher AE branch candidate features
            teacher_score:   [N]       teacher fused similarity score
            pos_mask:        [N]       positive sample mask (bool)
            neg_mask:        [N]       negative sample mask (bool)
        Returns:
            loss_total: scalar total loss
            loss_dict:  dict mapping each sub-loss to its value
        """
        loss_total = 0.0

        # 1. Response-level distillation (Response Distillation)
        # The student directly imitates the teacher's fused similarity score
        loss_response = F.mse_loss(student_score, teacher_score)
        loss_total += self.w_response * loss_response

        # 2. Feature-level distillation (Feature Distillation) - [temporarily disabled, testing response distillation only]
        # The student's 256-dim features imitate the 256-dim features of the teacher AE branch
        # Use a cosine similarity loss, focusing more on directional consistency
        # loss_feature = 1.0 - F.cosine_similarity(student_feat, teacher_feat_ae, dim=1).mean()
        # loss_total += self.w_feature * loss_feature
        loss_feature = 0.0

        # 3. Ranking distillation / tracking-aware distillation (Ranking Distillation) - [temporarily disabled, testing response distillation only]
        # Leverage labeled data: positive samples should have higher similarity than negative samples
        # loss_rank = 0.0
        # if pos_mask is not None and neg_mask is not None and pos_mask.any() and neg_mask.any():
        #     pos_scores = student_score[pos_mask]    # [N_pos]
        #     neg_scores = student_score[neg_mask]    # [N_neg]
        #
        #     # Compute the margin ranking loss over all positive-negative pairs
        #     # Expect pos_score > neg_score + margin
        #     # Equivalent to: max(0, margin - pos_score + neg_score)
        #     pos_scores = pos_scores.unsqueeze(1)    # [N_pos, 1]
        #     neg_scores = neg_scores.unsqueeze(0)    # [1, N_neg]
        #     ranking_loss = F.relu(self.margin - pos_scores + neg_scores)
        #     loss_rank = ranking_loss.mean()
        #     loss_total += self.w_rank * loss_rank
        loss_rank = 0.0

        return loss_total, {
            'loss_response': loss_response.item(),
            'loss_feature': loss_feature,
            'loss_rank': loss_rank
        }
