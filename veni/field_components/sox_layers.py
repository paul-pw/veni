# Code in part/adapted from https://github.com/lucidrains/VN-transformer/tree/main
from functools import wraps
from packaging import version
from collections import namedtuple
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helper

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# Attend

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        l2_dist = False
    ):
        super().__init__()
        assert not (flash and l2_dist), 'flash attention is not compatible with l2 distance'
        self.l2_dist = l2_dist

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major >= 8 and device_properties.minor >= 0:
            print_once('Flash-Attention-Capable GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non Flash-Attention-Capable GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale # b h q_len k_len

        # l2 distance

        if self.l2_dist:
            # -cdist squared == (-q^2 + 2qk - k^2)
            # so simply work off the qk above
            q_squared = reduce(q ** 2, 'b h i d -> b h i 1', 'sum')
            k_squared = reduce(k ** 2, 'b h j d -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v) # b h n d

        return out


# layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# equivariant modules

def check_axes(*axes):
    """
    checks if all dims exist in axes list
    returns False if there is an issue.
    """
    # flatten lists
    combined = [x for ax in axes for x in ax]
    combined_set = set(combined)
    if len(combined) != len(combined_set):
        # Axes contain duplicates.
        return False
    if combined_set != set(range(len(combined_set))):
        return False
    return True

class SOxLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        equivariant_axes,
        invariant_axes,
        invariant_axes_out = None,
        equivariant_axes_out = None, 
    ):
        super().__init__()

        self.equivariant_axes = equivariant_axes
        if equivariant_axes_out is not None:
            self.equivariant_axes_out = equivariant_axes_out
        else:
            self.equivariant_axes_out = equivariant_axes

        self.invariant_axes = invariant_axes
        if invariant_axes_out is not None:
            self.invariant_axes_out = invariant_axes_out
        else:
            self.invariant_axes_out = invariant_axes
        invariant_dim = len(invariant_axes)
        invariant_dim_out = len(self.invariant_axes_out)
        self.dim_coor_total_in = len(equivariant_axes) + len(invariant_axes)
        self.dim_coor_total_out = len(self.equivariant_axes_out) + len(self.invariant_axes_out)
        self.dim_out = dim_out

        assert check_axes(equivariant_axes, invariant_axes), "Axes are not valid"

        assert len(equivariant_axes) == len(self.equivariant_axes_out), "Equivariant dim cannot be changed!"
        assert max(self.invariant_axes_out+self.equivariant_axes_out) < self.dim_coor_total_out, "out indices cannot be larger than number of out axes"
        self.invariant_linear = nn.Sequential(
            nn.Flatten(-2,-1),
            nn.Linear(dim_in*(invariant_dim+1), dim_out*invariant_dim_out, bias=True),
            nn.Unflatten(-1, (dim_out,invariant_dim_out))
        )

        self.equivariant_weights = nn.Parameter(torch.empty(dim_out,invariant_dim+1,dim_in))
        k_sqrt = math.sqrt(1./(dim_in*(invariant_dim+1)+dim_out))
        nn.init.uniform_(self.equivariant_weights,a=-k_sqrt, b=k_sqrt)
        #nn.init.uniform_(self.linear.weight,a=-k_sqrt, b=k_sqrt)

    def forward(self, x):
        equivariant_x = x[...,self.equivariant_axes]
        invariant_x = x[...,self.invariant_axes]
        assert x.shape[-1] == self.dim_coor_total_in, "last dim must match configuration for invariant and equivariant axes"
            
        # Get Invariant feature of equivariant input 
        equi_to_invar_x = torch.linalg.vector_norm(equivariant_x, dim=-1, keepdim=True)
        # Invariant output calculation
        full_invariant_x = torch.cat([invariant_x,equi_to_invar_x], dim=-1)
        invariant_out = self.invariant_linear(full_invariant_x)      

        # Extend invariant features with ones for batch (for w_{11}+z_1*w_{12})
        extended_invar = torch.cat([torch.ones(invariant_x.shape[:-1], device=invariant_x.device).unsqueeze(-1), invariant_x], dim=-1)
        # i input_dims
        # k invariant_scalars + 1
        # o output_dims
        # v equivariant dims
        equivariant_out = torch.einsum("...iv,oki,...ik->...ov", equivariant_x,self.equivariant_weights, extended_invar)

        # Recombine equivariant and invariant to get new output.
        out = torch.empty((*x.shape[:-2], self.dim_out, self.dim_coor_total_out), device=x.device, dtype=x.dtype)
        out[...,self.equivariant_axes_out] = equivariant_out
        out[...,self.invariant_axes_out] = invariant_out
        return out


class VNReLU(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.U = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim = -1, keepdim = True).clamp(min = self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out

class SOxReLU(nn.Module):
    def __init__(
        self, 
        dim, 
        equivariant_axes,
        invariant_axes,
        eps = 1e-6):
        super().__init__()
        self.equivariant_axes = equivariant_axes
        self.invariant_axes = invariant_axes
        self.vn_relu = VNReLU(dim, eps)
        self.relu = nn.ReLU()

    def forward(self, x):
        equivariant_x = x[...,self.equivariant_axes]
        invariant_x = x[...,self.invariant_axes]
        equivariant_out = self.vn_relu(equivariant_x)
        invariant_out = self.relu(invariant_x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        out[...,self.equivariant_axes] = equivariant_out
        out[...,self.invariant_axes] = invariant_out
        return out


class SOxAttention(nn.Module):
    def __init__(
        self,
        dim,
        equivariant_axes, 
        invariant_axes,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash = False,
        num_latents = None   # setting this would enable perceiver-like cross attention from latents to sequence, with the latents derived from VNWeightedPool
    ):
        super().__init__()
        assert not (l2_dist_attn and flash), 'l2 distance attention is not compatible with flash attention'

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens = num_latents, squeeze_out_pooled_dim = False)

        self.to_q = SOxLinear(dim, dim_inner, equivariant_axes, invariant_axes)
        self.to_k = SOxLinear(dim, dim_inner, equivariant_axes, invariant_axes)
        self.to_v = SOxLinear(dim, dim_inner, equivariant_axes, invariant_axes)
        self.to_out = SOxLinear(dim_inner, dim, equivariant_axes, invariant_axes)

        if l2_dist_attn and not exists(num_latents):
            # tied queries and keys for l2 distance attention, and not perceiver-like attention
            self.to_k = self.to_q

        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn)

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        i - source sequence dimension
        j - target sequence dimension
        """

        c = x.shape[-1]

        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask = mask)
        else:
            q_input = x

        q, k, v = self.to_q(q_input), self.to_k(x), self.to_v(x) # b n (h * d) c where c = dim_coor + dim_feat
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n (d c)', h = self.heads), (q, k, v)) # b h n (d * c)

        out = self.attend(q, k, v, mask = mask) # b h n (d c)

        out = rearrange(out, 'b h n (d c) -> b n (h d) c', c = c) # b n (h * d) c
        return self.to_out(out)

def SOxFeedForward(dim, equivariant_axes, invariant_axes, mult = 4, bias_epsilon = 0.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        SOxLinear(dim, dim_inner, equivariant_axes, invariant_axes),
        SOxReLU(dim_inner, equivariant_axes, invariant_axes),
        #VNReLU(dim_inner, eps=1e-8),
        SOxLinear(dim_inner, dim, equivariant_axes, invariant_axes)
    )

class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -1)
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')

class VNWeightedPool(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        num_pooled_tokens = 1,
        squeeze_out_pooled_dim = True
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x, mask = None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim = 1)
            mean_pooled = numer / denom.clamp(min = 1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out

# equivariant VN transformer encoder

class SOxTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        equivariant_axes, 
        invariant_axes,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        ff_mult = 4,
        final_norm = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SOxAttention(dim = dim, 
                            equivariant_axes = equivariant_axes,
                            invariant_axes = invariant_axes,
                            dim_head = dim_head, 
                            heads = heads, bias_epsilon = bias_epsilon, l2_dist_attn = l2_dist_attn, flash = flash_attn),
                VNLayerNorm(dim),
                SOxFeedForward(dim, equivariant_axes, invariant_axes, mult = ff_mult),
                VNLayerNorm(dim)
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, 'input needs to be in the shape of (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask = mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)

# main class

class SOxTransformer(nn.Module):
    def __init__(
        self,
        latent_dim,
        equivariant_axes, 
        invariant_axes, # Feature axes are always assumed to be rotation invariant.
        depth=5,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        pool="mean", # If class Token is used "cls" to get the latents or Global Average Pooling.
        dim_coor = 3,
        patch_size = None,
        l2_dist_attn = False,
        flash_attn = False,
    ):
        super().__init__()

        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        self.dim_coor = dim_coor
        self.dim_coor_total = dim_coor + dim_feat

        # make feature axes always be rotation invariant (feats are alwys the last axes.)
        invariant_axes_feats = invariant_axes + list(range(dim_coor, dim_coor+dim_feat))
        self.invariant_axes = invariant_axes

        if not patch_size:
            self.vn_proj_in = nn.Sequential(
                Rearrange('... c -> ... 1 c'),
                SOxLinear(1, latent_dim,equivariant_axes, invariant_axes_feats, invariant_axes_out=invariant_axes)
            )
        else:
            self.vn_proj_in = nn.Sequential(
                #SOxLinear(patch_size, latent_dim,equivariant_axes, invariant_axes_feats),
                #SOxReLU(latent_dim,equivariant_axes, invariant_axes_feats),
                SOxLinear(patch_size,latent_dim, equivariant_axes,invariant_axes_feats, invariant_axes_out=invariant_axes)
            )

        self.pool = pool
        if self.pool == "cls":
            # SOx equivariant/invariant class token
            self.cls_inv = nn.Parameter(torch.zeros(1,1,latent_dim, len(invariant_axes)))
        
        self.encoder = SOxTransformerEncoder(
            dim = latent_dim,
            equivariant_axes = equivariant_axes, 
            invariant_axes = invariant_axes,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            dim_coor = self.dim_coor, # we decrease dim to dim_coor from dim_coor_total
            l2_dist_attn = l2_dist_attn,
            flash_attn = flash_attn
        )

    def forward(
        self,
        coors,
        feats,
        mask = None,
        return_concatted_coors_and_feats = False
    ):
        x = coors
        x = torch.cat((x, feats), dim=-1) # Shape: [Batch Size, Squence Length, 3+dim_feats]

        B = x.shape[0]
        C = x.shape[-1]
    
        assert C == self.dim_coor_total

        x = self.vn_proj_in(x) # Shape: [Batch Size, Sequence Lenght, Latent Dim, 3+dim_feats]
        
        if self.pool=="cls":
            self.cls_token = torch.zeros(1,1,x.shape[-2],self.dim_coor, device=x.device)
            self.cls_token[...,self.invariant_axes] = self.cls_inv
            cls_token = self.cls_token.expand(B,-1,-1,-1)
            x = torch.cat((cls_token, x), dim=1) # Shape: [Batch Size, Sequence Length + 1, Latent Dim, 3+dim_feats]

        # The actual Transformer Encoder.
        x = self.encoder(x, mask=mask)

        if self.pool == "cls":
            latent = x[:,0] # Shape: [Batch Size, Latent Dim, 3+dim_feats]
        elif self.pool == "mean":
            latent = x.mean(dim=1) # Shape: [Batch Size, Latent Dim, 3+dim_feats]
        else: 
            raise NotImplementedError()
        #latent_coor, latent_feat = latent[...,:self.dim_coor], latent[...,self.dim_coor:] # [Batch Size, Latent Dim, 3] [Batch Size, Latent Dim, dim_feats]
        return latent
