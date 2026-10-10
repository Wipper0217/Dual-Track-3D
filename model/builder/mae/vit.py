import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # -> (B, embed_dim, 8,8)
        x = x.flatten(2)  # -> (B, embed_dim, 64)
        x = x.transpose(1, 2)  # -> (B, 64, embed_dim)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)


    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class ViT(nn.Module):
    def __init__(self, in_chans=24, patch_size=3, embed_dim=128, depth=4, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)
        num_patches = (24 // patch_size) ** 2  # for patch=3 => 8x8=64

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # (B,64,dim)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,dim)
        x = torch.cat((cls, x), dim=1)  # (B,65,dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

if __name__ == "__main__":
    import torch
    x = torch.randn(16, 24, 24, 24)
    model = ViT()
    y = model(x)
    print(y.shape)