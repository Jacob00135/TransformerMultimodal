import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 2, dim // 1, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 1, dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim * 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim * 2, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(),
            nn.AvgPool3d(2, stride=2)
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        return conv4_out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class CrossTransformer_MOD_AVG(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'),
                                 nn.AdaptiveAvgPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'),
                                 nn.AdaptiveMaxPool1d(1),
                                 Rearrange('b d n -> b (d n)'))


    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=pet_tokens) + mri_tokens
            pet_tokens = pet_enc(pet_tokens, context=mri_tokens) + pet_tokens

        mri_cls_avg = self.gap(mri_tokens)
        mri_cls_max = self.gmp(mri_tokens)
        pet_cls_avg = self.gap(pet_tokens)
        pet_cls_max = self.gmp(pet_tokens)
        cls_token = torch.cat([mri_cls_avg,  pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        return cls_token


class model_ad(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # networks
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        # self.cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool3d(1), Rearrange('b c x y z -> b (c x y z)'))
        self.D = nn.Sequential(nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        alpha = torch.Tensor([2]).to(mri.device)
        # alpha = 1
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)
        # mri_embedding_vec = self.gap(mri_embeddings)
        # pet_embedding_vec = self.gap(pet_embeddings)

        # forward discriminator
        # mri_gt = torch.zeros([mri_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # pet_gt = torch.ones([pet_embeddings.shape[0], 1]).to(mri_embeddings.device)
        # mri_ad_loss = self.ad_loss(self.D(mri_embedding_vec), mri_gt)
        # pet_ad_loss = self.ad_loss(self.D(pet_embedding_vec), pet_gt)
        # D_loss = self.ad_loss(self.D(torch.cat([mri_embedding_vec, pet_embedding_vec], dim=0)), gt)
        # D_loss = (mri_ad_loss + pet_ad_loss)/2
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        # forward cross transformer
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        output_logits = self.fc_cls(output_pos)

        return output_logits, D_MRI_logits, D_PET_logits
