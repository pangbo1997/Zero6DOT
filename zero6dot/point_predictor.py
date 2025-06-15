import torch
import torch.nn as nn
from torch.nn import Module, Dropout


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

class TransLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(TransLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message




class PointPredictor(nn.Module):
    def __init__(
        self,
    ):
        super(PointPredictor, self).__init__()
        self.feat_transform=nn.Linear(6,128).cuda()
        self.self_attn_layer=TransLayer(256,8).cuda()
        self.cross_attn_layer=TransLayer(256,8).cuda()
        self.final_output=nn.Linear(256,3,bias=False).cuda()
    def forward(self,coords_unground,feats_unground,pts_ground,feats_ground,K):

        coords_unground=coords_unground[None].cuda().to(torch.float32)
        feats_unground=feats_unground[None].cuda().to(torch.float32)
        pts_ground=pts_ground[None].cuda().to(torch.float32)
        feats_ground=feats_ground[None].cuda().to(torch.float32)
        K=K[None].cuda().to(torch.float32)

        B=1
        center=pts_ground.mean(dim=1)
        scale=(pts_ground-center).norm(dim=-1).max(dim=-1)[0][:,None]

        unground_ground_match=torch.cdist(feats_unground,feats_ground).argmin(dim=-1)
        init_unground_z=pts_ground[:,:,2].reshape(-1)[unground_ground_match.reshape(-1)].reshape(B,-1)

        us,vs=torch.unbind(coords_unground,-1)
        zs = init_unground_z
        xs = (us - K[:,0, 2]) * zs / K[:,0, 0]
        ys = (vs - K[:,1, 2]) * zs / K[:,1, 1]
        input_pts_unground = torch.stack([xs, ys, zs], dim=-1)

        input_pts_unground=(input_pts_unground-center)/scale
        pts_ground=(pts_ground-center)/scale
        # import pdb;pdb.set_trace()
        f0_inputs=torch.cat([input_pts_unground,torch.zeros_like(input_pts_unground)],dim=-1)
        f0=self.feat_transform(f0_inputs.reshape(-1,6)).reshape(B,-1,128)
        f0=torch.cat([f0,feats_unground.detach()],dim=-1)

        f1_inputs=torch.cat([pts_ground,torch.ones_like(pts_ground)],dim=-1)
        f1=self.feat_transform(f1_inputs.reshape(-1,6)).reshape(B,-1,128)
        f1=torch.cat([f1,feats_ground.detach()],dim=-1)


        f0=self.self_attn_layer(f0,f0)
        f1=self.self_attn_layer(f1,f1)

        f0=self.cross_attn_layer(f0,f1)

        pred_pts_unground=input_pts_unground+self.final_output(f0)

        pred_pts_unground=pred_pts_unground*scale+center
        # import pdb;pdb.set_trace()
        return pred_pts_unground