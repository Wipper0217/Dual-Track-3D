"""
MoCo branch distillation loss V2: three-level joint distillation
- Feature-level distillation: cosine similarity loss between student projected features and teacher original features
- Response-level distillation: MSE loss between student NCC and teacher NCC
- Relation-level distillation: hinge loss preserving relative ranking across samples

Teacher: MoCo encoder_q (ResNet-50, frozen), outputs 2048-channel 3D feature maps
Student: StudentMoCoEncoder (ResNet-18), outputs 512-channel 3D feature maps + projection to 2048 channels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoDistillationLossV2(nn.Module):
    """
    MoCo branch distillation loss V2
    Three-level joint optimization: feature-level + response-level + relation-level

    L = α * L_feat + β * L_response + γ * L_relation

    where:
    - L_feat: cosine similarity loss between student projected features and teacher original features
    - L_response: MSE loss between student NCC and teacher NCC
    - L_relation: pairwise ranking loss preserving relative ranking across samples
    """
    def __init__(self, w_feat=1.0, w_response=0.5, w_relation=0.3, margin=0.1):
        """
        Args:
            w_feat: weight of the feature distillation loss (α)
            w_response: weight of the response distillation loss (β)
            w_relation: weight of the relation distillation loss (γ)
            margin: ranking margin in relation distillation
        """
        super().__init__()
        self.w_feat = w_feat
        self.w_response = w_response
        self.w_relation = w_relation
        self.margin = margin

    def _cosine_similarity_loss(self, f_student, f_teacher):
        """
        Feature-level distillation: cosine similarity loss
        Align student projected features with teacher original features

        Args:
            f_student: [B, 2048, D, H, W] student projected features
            f_teacher: [B, 2048, D, H, W] teacher original features
        Returns:
            loss: scalar, 1 - mean(cosine_similarity)
        """
        # L2 normalization along the channel dimension
        f_s_norm = F.normalize(f_student, p=2, dim=1)
        f_t_norm = F.normalize(f_teacher, p=2, dim=1)

        # Flatten to [B, C, L], where L = D*H*W
        B, C = f_s_norm.shape[0], f_s_norm.shape[1]
        f_s_flat = f_s_norm.view(B, C, -1)  # [B, C, L]
        f_t_flat = f_t_norm.view(B, C, -1)  # [B, C, L]

        # Compute cosine similarity along the channel dimension, giving [B, L]
        cos_sim = F.cosine_similarity(f_s_flat, f_t_flat, dim=1)  # [B, L]

        # Average over spatial positions, then average over the batch
        loss = 1.0 - cos_sim.mean()
        return loss

    def _response_distillation_loss(self, ncc_student, ncc_teacher):
        """
        Response-level distillation: NCC score MSE loss
        Make the student's NCC score approach the teacher's NCC score

        Args:
            ncc_student: [B] student NCC scores
            ncc_teacher: [B] teacher NCC scores
        Returns:
            loss: scalar
        """
        return F.mse_loss(ncc_student, ncc_teacher)

    def _relation_distillation_loss(self, ncc_student, ncc_teacher):
        """
        Relation-level distillation: preserve the relative ranking of positive and negative samples
        If the teacher considers A > B, the student should preserve this relation

        Uses pairwise ranking loss:
        For each pair (i, j), if teacher_i > teacher_j, then student_i > student_j is expected
        loss = mean(ReLU(margin - (s_i - s_j) * sign(t_i - t_j)))

        Args:
            ncc_student: [B] student NCC scores
            ncc_teacher: [B] teacher NCC scores
        Returns:
            loss: scalar
        """
        B = ncc_student.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=ncc_student.device)

        # Expand into pairs: [B, 1] - [1, B] = [B, B]
        s_i = ncc_student.unsqueeze(1)  # [B, 1]
        s_j = ncc_student.unsqueeze(0)  # [1, B]
        t_i = ncc_teacher.unsqueeze(1)  # [B, 1]
        t_j = ncc_teacher.unsqueeze(0)  # [1, B]

        # Teacher's relative relation
        teacher_diff = t_i - t_j  # [B, B]
        student_diff = s_i - s_j  # [B, B]

        # Only consider pairs where the teacher deems a significant difference
        mask = torch.abs(teacher_diff) > self.margin

        if mask.sum() == 0:
            return torch.tensor(0.0, device=ncc_student.device)

        # Ranking loss: if the teacher considers i > j, the student should also consider i > j
        target_sign = torch.sign(teacher_diff)
        loss = F.relu(self.margin - student_diff * target_sign)

        # Average over valid pairs only
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def forward(self, student_ncc, teacher_ncc,
                z_student=None, z_teacher=None,
                x_student=None, x_teacher=None):
        """
        Joint distillation loss

        Args:
            student_ncc: [B] student NCC scores
            teacher_ncc: [B] teacher NCC scores
            z_student: [B, 2048, D, H, W] student template projected features (optional)
            z_teacher: [B, 2048, D, H, W] teacher template original features (optional)
            x_student: [B, 2048, D, H, W] student candidate projected features (optional)
            x_teacher: [B, 2048, D, H, W] teacher candidate original features (optional)

        Returns:
            loss_total: scalar total loss
            loss_dict: dict, values of the individual loss terms
        """
        loss_total = torch.tensor(0.0, device=student_ncc.device)
        loss_dict = {}

        # 1. Feature-level distillation (when projected features are provided)
        if z_student is not None and z_teacher is not None:
            loss_feat_z = self._cosine_similarity_loss(z_student, z_teacher)
            loss_dict['loss_feat_z'] = loss_feat_z.item()

            if x_student is not None and x_teacher is not None:
                loss_feat_x = self._cosine_similarity_loss(x_student, x_teacher)
                loss_dict['loss_feat_x'] = loss_feat_x.item()
                loss_feat = (loss_feat_z + loss_feat_x) / 2
            else:
                loss_feat = loss_feat_z

            loss_dict['loss_feat'] = loss_feat.item()
            loss_total = loss_total + self.w_feat * loss_feat

        # 2. Response-level distillation
        loss_response = self._response_distillation_loss(student_ncc, teacher_ncc)
        loss_dict['loss_response'] = loss_response.item()
        loss_total = loss_total + self.w_response * loss_response

        # 3. Relation-level distillation
        loss_relation = self._relation_distillation_loss(student_ncc, teacher_ncc)
        loss_dict['loss_relation'] = loss_relation.item()
        loss_total = loss_total + self.w_relation * loss_relation

        loss_dict['loss_total'] = loss_total.item()

        return loss_total, loss_dict
