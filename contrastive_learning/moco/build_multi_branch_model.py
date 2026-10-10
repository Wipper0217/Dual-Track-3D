import torch
import torch.nn as nn
from builder_moco import MoCo3D
from model.model_3d.resnet.resnet_ae_3d import ResNetAutoEncoder3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dual-Branch Self-Supervised Tracking(DB-SST)
class DBSST(nn.Module):
    def __init__(self, autoencoder, moco):
        super(DBSST, self).__init__()
        self.ae = autoencoder
        self.moco = moco

    def forward(self, im_q, im_k):
        output = self.ae(im_q, mlp=True)
        logits, labels = self.moco(im_q, im_k)

        return {
            'ae': output,
            'moco': [logits, labels]
        }