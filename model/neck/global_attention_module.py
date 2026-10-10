import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.multi_head_attention_topk import MultiHeadAttentionTopK


class GlobalAttentionModule(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 num_layers=4,
                 dropout=0.1,
                 activation="relu",
                 f_z_size=15,
                 f_x_size=31):
        super(GlobalAttentionModule, self).__init__()
        layer = GlobalAttentionLayer(mid_channels_model=mid_channels_model,
                                     num_heads=num_heads,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     activation=activation)
        self.d_model = mid_channels_model
        self.layers = _get_clones(layer, num_layers)
        self.f_z_size = f_z_size
        self.f_x_size = f_x_size
        self.fz_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_z_size)
        self.fx_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_x_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'pos' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f_z, f_x):
        pos_z = self.fz_pos_encoding(f_z)   # (B, C, H, W)
        pos_x = self.fx_pos_encoding(f_x)   # (B, C, H, W)
        f_z = f_z.flatten(2).permute(2, 0, 1).contiguous()
        pos_z = pos_z.flatten(2).permute(2, 0, 1).contiguous()
        f_x = f_x.flatten(2).permute(2, 0, 1).contiguous()
        pos_x = pos_x.flatten(2).permute(2, 0, 1).contiguous()
        output_z = f_z
        output_x = f_x

        for layer in self.layers:
            output_z, output_x = layer(output_z, output_x, pos_z, pos_x)
        output_x = output_x.permute(1, 2, 0).contiguous()
        output_x = output_x.view(*output_x.shape[:2], self.f_x_size, self.f_x_size)
        return output_x


class MultiGlobalAttentionModule(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 num_layers=4,
                 dropout=0.1,
                 activation="relu",
                 f_z_size=8,
                 f_x_size=16):
        super(MultiGlobalAttentionModule, self).__init__()
        layer_1 = GlobalAttentionLayer(mid_channels_model=mid_channels_model,
                                     num_heads=num_heads,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     activation=activation)
        layer_2 = GlobalAttentionLayer(mid_channels_model=mid_channels_model,
                                     num_heads=num_heads,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     activation=activation)
        self.d_model = mid_channels_model
        self.layers_1 = _get_clones(layer_1, num_layers)
        self.layers_2 = _get_clones(layer_2, num_layers)
        self.f_z_size = f_z_size
        self.f_x_size = f_x_size
        self.f_y_size = f_z_size
        self.fz_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_z_size)
        self.fx_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_x_size)
        self.fy_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_y_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'pos' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f_z, f_x, f_y):
        pos_z = self.fz_pos_encoding(f_z)  # (B, C, H, W)
        pos_x = self.fx_pos_encoding(f_x)  # (B, C, H, W)
        pos_y = self.fy_pos_encoding(f_y)
        f_z = f_z.flatten(2).permute(2, 0, 1).contiguous()
        pos_z = pos_z.flatten(2).permute(2, 0, 1).contiguous()
        f_x = f_x.flatten(2).permute(2, 0, 1).contiguous()
        pos_x = pos_x.flatten(2).permute(2, 0, 1).contiguous()
        output_z = f_z
        output_x = f_x


class GlobalAttentionModuleV2(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 num_layers=4,
                 dropout=0.1,
                 activation="relu",
                 f_z_size=15,
                 f_x_size=31,
                 top_k=32):
        super(GlobalAttentionModuleV2, self).__init__()
        layer = GlobalAttentionLayerV2(mid_channels_model=mid_channels_model,
                                     num_heads=num_heads,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     activation=activation,
                                     top_k=top_k)
        self.decoder = DecoderLayer(d_model=mid_channels_model,
                                    nhead=num_heads,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout,
                                    activation=activation)
        self.d_model = mid_channels_model
        self.layers = _get_clones(layer, num_layers)
        self.f_z_size = f_z_size
        self.f_x_size = f_x_size
        self.fz_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_z_size)
        self.fx_pos_encoding = SpatialPositionEncodingLearned(self.d_model, self.f_x_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, f_z, f_x):
        pos_z = self.fz_pos_encoding(f_z)   # (B, C, H, W)
        pos_x = self.fx_pos_encoding(f_x)   # (B, C, H, W)

        f_z = f_z.flatten(2).permute(2, 0, 1).contiguous()
        pos_z = pos_z.flatten(2).permute(2, 0, 1).contiguous()
        f_x = f_x.flatten(2).permute(2, 0, 1).contiguous()
        pos_x = pos_x.flatten(2).permute(2, 0, 1).contiguous()

        output_z = f_z
        output_x = f_x

        for layer in self.layers:
            output_z, output_x = layer(output_z, output_x, pos_z, pos_x)
        output_x = self.decoder(output_z, output_x)
        output_x = output_x.permute(1, 2, 0).contiguous()
        output_x = output_x.view(*output_x.shape[:2], self.f_x_size, self.f_x_size)
        return output_x

class GlobalAttentionLayer(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(GlobalAttentionLayer, self).__init__()
        assert mid_channels_model % num_heads == 0
        self.encoder_layer = EncoderLayer(d_model=mid_channels_model,
                                          nhead=num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)
        self.decoder_layer = DecoderLayer(d_model=mid_channels_model,
                                          nhead=num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, scr2, pos_src1, pos_src2, src1_mask=None, src2_mask=None):
        src1_enc = self.encoder_layer(src1, pos_src1, src1_mask)
        src2_enc = self.encoder_layer(scr2, pos_src2, src2_mask)
        dec_output = self.decoder_layer(src1_enc, src2_enc, pos_src1, pos_src2, src1_mask, src2_mask)
        return src1_enc, dec_output


class GlobalAttentionLayerV2(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 top_k=32):
        super(GlobalAttentionLayerV2, self).__init__()
        assert mid_channels_model % num_heads == 0
        self.template_encoder_layer = EncoderLayer(d_model=mid_channels_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        self.search_encoder_layer = SparseEncoderLayer(d_model=mid_channels_model,
                                                       nhead=num_heads,
                                                       dim_feedforward=dim_feedforward,
                                                       activation=activation,
                                                       top_k=top_k)
        self.decoder_layer = DecoderLayer(d_model=mid_channels_model,
                                          nhead=num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, scr2, pos_src1, pos_src2, src1_mask=None, src2_mask=None):
        src1_enc = self.template_encoder_layer(src1, pos_src1, src1_mask)
        src2_enc = self.search_encoder_layer(scr2, pos_src2, src2_mask)
        dec_output = self.decoder_layer(src1_enc, src2_enc, pos_src1, pos_src2, src1_mask, src2_mask)
        return src1_enc, dec_output





class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos_src=None, src_mask=None):
        q = k = self.with_pos_embed(src, pos_src)
        src2 = self.self_attn(query=q, key=k, value=src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SparseEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", top_k=32):
        super(SparseEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttentionTopK(embed_dim=d_model, num_heads=nhead, dropout=dropout, top_k=top_k)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos_src=None, src_mask=None):
        q = k = self.with_pos_embed(src, pos_src)
        src2 = self.slf_attn(query=q, key=k, value=src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, pos_src1=None, pos_src2=None, src1_mask=None, src2_mask=None):
        """

        Args:
            src1: template features
            src2: search features
            pos_src1: template features position encoding
            pos_src2: search features position encoding
        Returns:
            output: search features
        """
        q = self.with_pos_embed(src2, pos_src2)
        k = self.with_pos_embed(src1, pos_src1)
        output2 = self.cross_attn(query=q, key=k, value=src1, attn_mask=src1_mask)[0]
        output = src2 + self.dropout1(output2)
        output = self.norm1(output)

        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        return output

class LongAndShortAttentionModule(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 num_layers=4,
                 dropout=0.1,
                 activation="relu",
                 f_z_size=15,
                 f_x_size=31):
        self.long_term_attn = GlobalAttentionModule(
            mid_channels_model=mid_channels_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            f_z_size=f_z_size,
            f_x_size=f_x_size
        )
        self.short_term_attn = GlobalAttentionModule(
            mid_channels_model=mid_channels_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            f_z_size=f_z_size,
            f_x_size=f_x_size
        )
        self.linear1 = nn.Linear(mid_channels_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, mid_channels_model)

        self.norm1 = nn.LayerNorm(mid_channels_model)
        self.norm2 = nn.LayerNorm(mid_channels_model)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, f_z, f_x, f_y):
        long_tgt = self.long_term_attn(f_z, f_x)
        short_tgt = self.short_term_attn(f_y, f_x)
        tgt = f_x + self.dropout1_1(long_tgt) + self.dropout1_2(short_tgt)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


class SpatialPositionEncodingLearned(nn.Module):
    def __init__(self, d_model, score_size):
        super(SpatialPositionEncodingLearned, self).__init__()
        self.row_embed = nn.Embedding(score_size, d_model // 2)
        self.col_embed = nn.Embedding(score_size, d_model // 2)
        self.spatial_size = score_size
        self.pos = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def get_position_encoding(self, x):
        i = torch.arange(self.spatial_size, device=x.device)
        j = torch.arange(self.spatial_size, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(self.spatial_size, 1, 1),
            y_emb.unsqueeze(1).repeat(1, self.spatial_size, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1, C, H, W
        return pos

    def forward(self, x):
        return self.get_position_encoding(x)    # (B, C, H, W)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

if __name__ == '__main__':
    mid_channels = 256
    f_z_size = 15
    f_x_size = 31
    B = 4
    zf = torch.randn(B, mid_channels, f_z_size, f_z_size)
    xf = torch.randn(B, mid_channels, f_x_size, f_x_size)
    gam = GlobalAttentionModule(mid_channels, f_z_size=f_z_size, f_x_size=f_x_size)
    gamv2 = GlobalAttentionModuleV2(mid_channels, f_z_size=f_z_size, f_x_size=f_x_size)
    output = gam(zf, xf)
    output_2 = gamv2(zf, xf)
    print(output.shape)
    print(output_2.shape)
