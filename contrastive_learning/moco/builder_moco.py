import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MoCo3D(nn.Module):
    def __init__(self, base_encoder, input_dim, mlp_dim, output_dim, K=65536, m=0.999, T=0.07, mlp=False):
        """

        :param base_encoder: base encoder
        :param input_dim: output dimension of the encoder
        :param mlp_dim: mlp dimension
        :param output_dim: output dimension, used for contrastive learning
        :param K:
        :param m:
        :param T:
        :param mlp:
        """
        super(MoCo3D, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        if mlp:
            self.encoder_q.fc = self._build_mlp(input_dim, mlp_dim, output_dim)
            self.encoder_k.fc = self._build_mlp(input_dim, mlp_dim, output_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(output_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_mlp(self, input_dim, mlp_dim, output_dim):
        mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )
        return mlp

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x
        batch_size_all = x_gather.shape[0]

        # random shuffle index
        if torch.cuda.is_available():
            idx_shuffle = torch.randperm(batch_size_all).cuda()
        else:
            idx_shuffle = torch.randperm(batch_size_all)


        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x_gather[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x
        batch_size_all = batch_size_this

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = 0
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features

        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # if torch.cuda.is_available():
        #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # else:
        #     labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels