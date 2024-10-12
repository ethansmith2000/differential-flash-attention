import math
from einops import rearrange
import torch
import torch.nn.functional as F
import math
from torch import nn


class GroupRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, num_groups=1, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.num_groups = num_groups
        assert dim % num_groups == 0, 'dim must be divisible by num_groups'

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = x.reshape(*x.shape[:-1], self.num_groups, -1)
        output = self._norm(x.float()).type_as(x)
        output = output.reshape(*output.shape[:-2], -1)
        if self.weight is not None:
            output = output * self.weight
        return output



def lambda_init_fn(depth):
    """ create a [0,1] lambda init value based on depth in the paper """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def prep_qkv(x):
    """ 
    go from double dim to double heads 
    b, nh, seq, d*2 -> b nh*2 seq d
    """
    x = rearrange(x, 'b nh seq (n d) -> b nh seq n d', n=2)
    x = rearrange(x, 'b nh seq n d -> b (nh n) seq d').contiguous()
    return x


def vanilla_diff_attention(q, k, v, lmbda, causal=True, mask=None):
    # go from double dim to double heads
    q, k = map(prep_qkv, (q, k))
    nh = q.shape[1] // 2

    # compute softmaxed attention matrix
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    # masking
    if causal:
        offset = q.shape[-2] - k.shape[-2]
        mask = torch.triu(
                torch.zeros([q.shape[-2], k.shape[-2]]).float()
                .fill_(float("-inf"))
                .type_as(scores),
                1 + offset,
            )
    if mask is not None:
        scores += mask[None, None]


    attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)

    # do the subtraction
    attn = attn[:, :nh] - attn[:, nh:] * lmbda

    # multiply by value
    output = torch.matmul(attn, v)

    return output


def flash_diff_attention(q, k, v, lmbda, causal=True, mask=None):
    # go from double dim to double heads
    q, k = map(prep_qkv, (q, k))
    nh = q.shape[1] // 2

    # need to dupe our values for parallelization
    out = F.scaled_dot_product_attention(q, k, torch.cat((v, v), dim=1), is_causal=causal, attn_mask=mask)

    # take difference of output values
    out = out[:, :nh] - out[:, nh:] * lmbda

    return out



class DifferentialAttention(torch.nn.Module):
    """
    from the paper: https://arxiv.org/pdf/2410.05258
    """

    def __init__(self, 
                dim=512, 
                context_dim=512, 
                heads=8, 
                layer_num=1, 
                flash=False,
                causal=True,
                ):
        super().__init__()
        self.l_q1, self.l_k1, self.l_q2, self.l_k2 = (nn.Parameter(torch.randn(heads) * 0.1) for _ in range(4))
        self.wq = torch.nn.Linear(dim, dim * 2, bias=False)
        self.wk = torch.nn.Linear(context_dim, dim * 2, bias=False)
        self.wv = torch.nn.Linear(context_dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
        self.l_init = torch.Tensor([lambda_init_fn(layer_num)])
        self.norm = GroupRMSNorm(dim, num_groups=heads)
        self.causal = causal

        self.attn_fn = flash_diff_attention if flash else vanilla_diff_attention

    def forward(self, x, context=None, mask=None):
        # suppport cross attention
        context = context or x

        # lambda calculation
        lambda_1 = torch.exp(torch.sum(self.l_q1 * self.l_k1, dim=-1).float()).type_as(x)
        lambda_2 = torch.exp(torch.sum(self.l_q2 * self.l_k2, dim=-1).float()).type_as(x)
        lmbda = lambda_1 - lambda_2 + self.l_init

        # project, permute to head dimension, and do attention
        q, k, v = self.wq(x), self.wk(context), self.wv(context)
        q, k, v = map(lambda t: rearrange(t, 'b seq (nh d) -> b nh seq d', nh=8), (q, k, v))
        out = self.attn_fn(q, k, v, lmbda[None, :, None, None], causal=self.causal, mask=mask)

        # RMSnorm and the lambda scaling
        out = self.norm(out) * (1 - self.l_init)
        
        # back to b,s,d
        out = rearrange(out, 'b nh seq d -> b seq (nh d)')

        return self.wo(out)



class ExperimentalDifferentialAttention(torch.nn.Module):
    """
    allows some experimental modifications, like learned lambda terms per-head, per-dim, or both
    rms affine is also supported
    and option for bias
    """

    def __init__(self, 
                dim=512, 
                context_dim=512, 
                heads=8, 
                layer_num=1, 
                flash=False,
                causal=True,
                lambda_mode = "uniform", # "uniform", "per_head", "per_dim", "per_head_and_dim"
                rms_affine = False,
                out_bias=False
                ):
        super().__init__()
        self.l_init = torch.Tensor([lambda_init_fn(layer_num)])

        lmbda = torch.Tensor([lambda_init_fn(layer_num)])
        if lambda_mode == "uniform":
            self.lmbda = torch.nn.Parameter(lmbda[None,:,None,None])
        elif lambda_mode == "per_head":
            self.lmbda = torch.nn.Parameter(torch.ones(1, heads, 1, 1) * lmbda)
        elif lambda_mode == "per_dim":
            assert flash, "per_dim only supported with flash"
            self.lmbda = torch.nn.Parameter(torch.ones(1, 1, 1, dim // heads) * lmbda)
        elif lambda_mode == "per_head_and_dim":
            assert flash, "per_head_and_dim only supported with flash"
            self.lmbda = torch.nn.Parameter(torch.ones(1, heads, 1, dim // heads) * lmbda)
        else:
            raise ValueError(f"lambda_mode {lambda_mode} not recognized")

        self.wq = torch.nn.Linear(dim, dim * 2, bias=False)
        self.wk = torch.nn.Linear(context_dim, dim * 2, bias=False)
        self.wv = torch.nn.Linear(context_dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=out_bias)
        self.l_init = torch.Tensor([lambda_init_fn(layer_num)])
        self.norm = GroupRMSNorm(dim, num_groups=heads, elementwise_affine=rms_affine)
        self.causal = causal

        self.attn_fn = flash_diff_attention if flash else vanilla_diff_attention


    def forward(self, x, context=None, mask=None):
        # suppport cross attention
        context = context or x

        # project, permute to head dimension, and do attention
        q, k, v = self.wq(x), self.wk(context), self.wv(context)
        q, k, v = map(lambda t: rearrange(t, 'b seq (nh d) -> b nh seq d', nh=8), (q, k, v))
        out = self.attn_fn(q, k, v, self.lmbda, causal=self.causal, mask=mask)

        # RMSnorm and the lambda scaling
        out = self.norm(out) * (1 - self.l_init)
        
        # back to b,s,d
        out = rearrange(out, 'b nh seq d -> b seq (nh d)')

        return self.wo(out)



if __name__ == "__main__":
    dim = 512

    # create our model
    model = DifferentialAttention(dim=dim, context_dim=dim, heads=8, layer_num=1, flash=False).requires_grad_(False)
    x = torch.randn(1, 32, dim)


    vanilla_out = model(x)
    model.attn_fn = flash_diff_attention
    flash_out = model(x)

    print(torch.allclose(vanilla_out, flash_out, atol=1e-4))


    model = ExperimentalDifferentialAttention(dim=dim, context_dim=dim, heads=8, layer_num=1, flash=False, lambda_mode="per_head").requires_grad_(False)
    x = torch.randn(1, 32, dim)

    vanilla_out = model(x)
    model.attn_fn = flash_diff_attention
    flash_out = model(x)

    print(torch.allclose(vanilla_out, flash_out, atol=1e-4))


