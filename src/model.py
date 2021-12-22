import copy
import typing
import math

import numpy as np
import revlib
import torch
import torch.utils.data
from deepspeed.runtime import lr_schedules
from torch.nn import functional as F

from src.dataclass import Context
from src.optimizers.build import build_optimizer

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter, typing.List[int]], gain: float):
    original_input = inp
    if isinstance(inp, list):
        inp = torch.zeros(inp)
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
    if isinstance(original_input, list):
        return torch.nn.Parameter(inp)
    return original_input

def init_(t, dim = None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1. / math.sqrt(dim)
    return torch.nn.init.normal_(t, mean=0, std=std)


class TripleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale0: torch.Tensor, scale1: torch.Tensor, shift: torch.Tensor, norm_power: int):
        # linear_attention chunk names:
        #   scale0 = depth, scale1 = scale, shift = shift
        scale0_relu = scale0.relu()
        inp = scale0_relu.pow(3) * scale1 + shift
        inp = inp - inp.mean(1, True)
        rstd = inp.size(1) ** (1 / norm_power) / inp.norm(norm_power, 1, True)
        inp *= rstd
        if scale1.requires_grad:
            ctx.save_for_backward(scale0_relu, scale1, inp, rstd)
        return inp

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None, None
        scale0_relu, scale1, out, rstd = ctx.saved_tensors
        dout = dout * rstd
        dout -= (dout * out).mean(1, True) * out
        dout -= dout.mean(1, True)
        d_scale = dout * scale0_relu.square()
        return d_scale * scale1 * 3, d_scale * scale0_relu, dout, None


def conv(inp: torch.Tensor, weight: torch.Tensor, groups: int, use_pad: bool) -> torch.Tensor:
    if use_pad and weight.size()[-1] - 1 > 0:
        inp = F.pad(inp, (weight.size()[-1] - 1, 0))
    return F.conv1d(inp, weight, groups=groups)


def expert_matmul(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgf,gfo->bgo", inp, weight)


class AuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()


def rnoise(params, zero, x):
    N, c, d = x.shape
    A, b, alpha, r = params
    mu = x.sum(1, keepdim=True)
    mu_mean = mu.sum(dim=(1),keepdim=True)*(1/c)
    s = mu - mu_mean
    s = s / torch.abs(s).max()
    sd = A * s + b
    s = alpha*sd + (1 - alpha) + 1
    sigma = s / torch.linalg.vector_norm(s)
    out = r * sigma * x + r * sigma * zero.repeat(x.shape).normal_()
    return out

def moe(inp: torch.Tensor, expert_weights: torch.nn.ParameterList, r: typing.Optional[torch.nn.ParameterList], zero: typing.Optional[torch.Tensor], training: bool,
        jitter_epsilon: float, feature_shuffle: torch.Tensor, groups: int, experts: int, model_noise: bool) -> torch.Tensor:
    *expert_weights, gate = expert_weights
    batch, features, sequence = inp.size()
    tokens = batch * sequence
    capacity = tokens // experts

    # get gates
    if gate.dtype != torch.float32:
        gate = gate.float()
    input_fp32 = inp.float()
    if training and model_noise:
        input_fp32 = rnoise(r, zero, input_fp32)
    elif training:
        input_fp32 = input_fp32 * (torch.rand_like(input_fp32) * jitter_epsilon + 1)
    inp = input_fp32.transpose(1, 2).reshape(tokens, features)

    #matrix multiplication to find tokens' most similar expert
    logits = inp.mm(gate)
    gates = F.softmax(logits, dim=1)

    # calculate permutation/ assign experts
    with torch.no_grad():
        mask = torch.ones_like(gates[:, 0])
        out = []
        for g in gates.unbind(1):
            _, idx = torch.topk(g * mask, capacity, 0)
            out.append(idx)
            mask[idx] = 0
        expert_permutation = torch.stack(out, 1)
        expert_permutation = expert_permutation.view(-1, 1).long()
        permutation_inverse = torch.argsort(expert_permutation, 0).view(-1, 1)
        expert_index = permutation_inverse // capacity

    # apply loss
    AuxLoss(gates.sum() / tokens)
    inp = inp * gates.gather(1, expert_index)

    # permute
    inp = inp.gather(0, expert_permutation.expand_as(inp))

    if feature_shuffle is not None:
        inp = inp.gather(1, feature_shuffle.view(1, -1).expand_as(inp))
    inp = inp.view(tokens // experts, experts * groups, features // groups)
    if len(expert_weights) == 1:
        inp = expert_matmul(inp, expert_weights[0])
    else:
        inp = torch.cat([expert_matmul(c, w) for c, w in zip(inp.chunk(len(expert_weights), 1), expert_weights)], -1)
    inp = inp.reshape(tokens, -1)
    inp = inp.gather(0, permutation_inverse.view(-1, 1).expand_as(inp))
    inp = inp.view(batch, sequence, -1).transpose(1, 2)
    return inp


def moe_check(inp: torch.Tensor, w: torch.nn.ParameterList, r: typing.Optional[torch.nn.ParameterList], zero: typing.Optional[torch.Tensor], training: bool,
              jitter_epsilon: float, feature_shuffle: torch.Tensor, groups: int, experts: int, model_noise: bool) -> torch.Tensor:
    if experts > 0:
        return moe(inp, w, r, zero, training, jitter_epsilon, feature_shuffle, groups, experts, model_noise)
    return conv(inp, w[0], groups, False)


def linear_attention(inp: torch.Tensor, divisor: torch.Tensor,
                     w0: typing.Union[torch.nn.ParameterList, torch.nn.Parameter], r0: typing.Optional[torch.nn.ParameterList],
                     feature_shuffle0: typing.Optional[torch.Tensor], groups0: int, experts0: int,
                     w1: torch.Tensor, r1: typing.Optional[torch.nn.ParameterList], w2: torch.nn.ParameterList, zero: typing.Optional[torch.Tensor],
                     feature_shuffle2: typing.Optional[torch.Tensor], groups2: int, experts2: int,
                     input_cache: torch.Tensor, cumsum_cache: torch.Tensor, bottleneck_group: int, training: bool,
                     caching: bool, idx: int, norm_power: int, jitter_epsilon: float,
                     pkm_layer: bool, pkm_keys: torch.nn.Parameter, pkm_values: typing.Optional[torch.nn.EmbeddingBag], input_dropout: typing.Optional[torch.nn.Dropout],
                     query_dropout: typing.Optional[torch.nn.Dropout], value_dropout: typing.Optional[torch.nn.Dropout],
                     pkm_topk: int, num_keys: int, pkm_heads: int, norm: typing.Optional[torch.nn.BatchNorm1d], model_noise: bool
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: Fix kernel_size back to being dynamic
    kernel_size = 7
    pad = True
    if not training and caching:
        if idx - 1 > kernel_size and inp.size(2) == 1:
            pad = False
            inp = torch.cat([input_cache, inp], -1)
        input_cache = inp[:, :, -kernel_size + 1:].detach()
    # w0 and w2 = moe params
    #input projection to (intermediaries * 3)
    #input dims = batch, features, sequence

    # featues -> intermediate * 3
    # inp.shape = (batch, features * 3, sequence)
    inp = moe_check(inp, w0, r0, zero, training, jitter_epsilon, feature_shuffle0, groups0, experts0, model_noise)
    #split projected tensor into three, each with orig. intermediary size
    depth, scale, shift = inp.chunk(3, 1)
    cum = depth.cumsum(-1)
    if not training and caching:
        cum = cum + cumsum_cache
        scale = scale[:, :, -1:]
        shift = shift[:, :, -1:]
        cum = cum[:, :, -1:]
        if idx - 1 > kernel_size:
            cumsum_cache = cum.detach()
    # intermediate * 3 -> intermediate
    inp = TripleNorm.apply(cum / divisor, scale, shift, norm_power)
    if pkm_layer:
        inp = conv(inp, w1, groups2, True)
        inp = inp.transpose(2,1)
        inp = pkm(inp, w2, pkm_keys, pkm_values, input_dropout, query_dropout, value_dropout, pkm_topk, num_keys, pkm_heads, norm)
        inp = inp.transpose(2,1)
    else:
        # intermediate -> intermediate * 3
        inp = conv(inp, w1, bottleneck_group, pad)
        # intermediate * 3 -> intermediate
        inp = TripleNorm.apply(*inp.chunk(3, 1), norm_power)
        # intermediate -> features
        inp = moe_check(inp, w2, r1, zero, training, jitter_epsilon, feature_shuffle2, groups2, experts2, False)
    return input_cache, cumsum_cache, inp

def pkm(inp: torch.Tensor, to_queries: torch.nn.Parameter, pkm_keys: torch.nn.Parameter,
        pkm_values: torch.nn.EmbeddingBag, input_dropout: torch.nn.Dropout,
        query_dropout:torch.nn.Dropout, value_dropout: torch.nn.Dropout, pkm_topk: int,
        num_keys: int, heads: int, norm: torch.nn.BatchNorm1d, splits: int = 2):  # add this as a param later on
    b, t, e, h = *inp.shape, heads
    inp = input_dropout(inp)
    queries = F.linear(inp,to_queries)
    queries = norm(queries)
    queries = query_dropout(queries)
    
    queries = queries.view(b, t, h, -1, 2)
    assignment = torch.einsum('bthdp,hnpd->bthpn', queries, pkm_keys)
    assignment = assignment - assignment.max((-2, -1)).values
    assignment = assignment.exp()
    normalizer = dots.sum(-1).prod(-1)
    scores, indices = dots.max(-2)
    attn = scores.sum(-1) / normalizer
    
    indices = indices * num_keys ** torch.arange(splits, device=indices.device, dtype=indices.dtype).view(1, 1, 1, -1)
    indices = indices.sum(-1)
    
    indices, attn = map(lambda x: x.reshape(-1, h), (indices, attn))

    out = pkm_values(indices, per_sample_weights=attn)
    out = value_dropout(out)
    return out.reshape(b, t, e)

# w1 inputs:
# conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
#                              ctx.model.activation_std)
def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)


class Trainer(torch.nn.Module):
    def __init__(self, ctx: Context, model: torch.nn.Module, data: typing.Optional[torch.Tensor]):
        super(Trainer, self).__init__()
        self.ctx = ctx
        self.model = torch.jit.trace(model, data) if data else model
        self.optimizer = build_optimizer(ctx, self.model.parameters())
        self.scheduler = lr_schedules.OneCycle(self.optimizer,
                                               ctx.optimizer.one_cycle.cycle_min_lr,
                                               ctx.optimizer.one_cycle.cycle_max_lr,
                                               ctx.optimizer.one_cycle.decay_lr_rate,
                                               ctx.optimizer.one_cycle.cycle_first_step_size,
                                               ctx.optimizer.one_cycle.cycle_second_step_size,
                                               ctx.optimizer.one_cycle.cycle_first_stair_count,
                                               ctx.optimizer.one_cycle.cycle_second_stair_count,
                                               ctx.optimizer.one_cycle.decay_step_size,
                                               ctx.optimizer.one_cycle.cycle_momentum,
                                               ctx.optimizer.one_cycle.cycle_min_mom,
                                               ctx.optimizer.one_cycle.cycle_max_mom,
                                               ctx.optimizer.one_cycle.decay_mom_rate,
                                               ctx.optimizer.one_cycle.last_batch_iteration)

    @torch.no_grad()
    def _to_device_detach(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.to(device=self.ctx.model.device, non_blocking=True).detach()

    def _forward_backward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(self.model(self._to_device_detach(src)), self._to_device_detach(tgt))
        loss.backward()
        return loss.detach()

    @torch.no_grad()
    def _clip_gradient(self):
        for p in self.gradients():
            g_norm = p.grad.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.zero_division_eps)
            p_norm = p.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.eps)
            grad_scale = (p_norm / g_norm * self.ctx.optimizer.agc.gradient_clipping).clamp(max=1)
            p.grad.data.copy_(p.grad * grad_scale)

    def accumulated_step(self, data: torch.Tensor) -> torch.Tensor:
        loss = sum(self._forward_backward(s, t) for s, t in zip(*data))
        self._clip_gradient()
        return loss

    @torch.no_grad()
    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    @torch.no_grad()
    def gradients(self) -> torch.nn.Parameter:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            yield p

    def save(self):
        torch.save(self.state_dict(), self.ctx.model.checkpoint_path)

    def load(self):
        wrong_keys = self.load_state_dict(torch.load(self.ctx.model.checkpoint_path), strict=False)
        for key in wrong_keys.missing_keys + wrong_keys.unexpected_keys:
            if not any(k.startswith('_') for k in key.split('.')):
                if key in wrong_keys.missing_keys:
                    raise ValueError(f"{key} is missing in checkpoint but exists in model")
                if key in wrong_keys.unexpected_keys:
                    raise ValueError(f"{key} is missing in model but exists in checkpoint")


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor):
        return inp * self.beta


class LinearAttention(torch.nn.Module):
    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()
        self.embedding = torch.nn.Embedding(ctx.dataset.classes, ctx.model.features * 2).to(ctx.model.device)
        orthonormal(self.embedding.weight, ctx.model.input_embedding_std * 2 ** -0.5)

        pos_embd = torch.arange(0, ctx.model.sequence_length).unsqueeze(0) + 1
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float).to(ctx.model.device))

        cell = LinearAttentionCell(self, ctx, 1)
        self.stem = revlib.ReversibleSequential(*[c
                                                  for i in range(1, 1 + ctx.model.depth)
                                                  for c in [cell.momentum((1 - ctx.model.momentumnet_beta) /
                                                                          ctx.model.momentumnet_beta ** i, not ctx.model.weight_sharing, i),
                                                            MomentumNetSide(ctx.model.momentumnet_beta ** i)]],
                                                target_device=ctx.model.device)
        self.output = torch.nn.Conv1d(ctx.model.features * 2, ctx.dataset.classes, (1,)).to(ctx.model.device)
        torch.nn.init.zeros_(self.output.weight.data)

    def forward(self, inp: torch.Tensor):
        return self.output(self.stem(self.embedding(inp).transpose(1, 2)))

    def reset_cache(self):
        for mod in self.stem.modules():
            if isinstance(mod, LinearAttentionCell):
                mod.reset_cache()


class MaskedBatchNorm1D(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, mask = None):
        b, t, d = x.shape
        has_mask = mask is not None

        if has_mask:
            initial_x = x
            mask = mask.unsqueeze(-1)
            x = x.masked_select(mask)

        shape = x.shape
        x = x.reshape(-1, d)
        x = self.fn(x)
        x = x.reshape(*shape)

        if has_mask:
            x = initial_x.masked_scatter(mask, x)

        return x


class ParameterStore(torch.nn.Module):
    """
    Something (likely deepspeed) changes all parameters in a ParameterList to [1] even though standalone parameters
    work. That's why a torch.nn.ModuleList of ParameterStores needs to be initialized.
    """

    def __init__(self, param: torch.Tensor):
        super(ParameterStore, self).__init__()
        self.param = torch.nn.Parameter(param)

    def __repr__(self):
        return (f'{self.__class__.__name__}(shape={str(list(self.param.size()))}, device={self.param.device}, '
                f'dtype={self.param.dtype})')

def get_riemann_noise_params(size):
    params = []
    params.append(torch.nn.Parameter(torch.rand(1, size)))
    params.append(torch.nn.Parameter(torch.rand(1, )))
    params.append(torch.nn.Parameter(torch.rand(1, )))
    params.append(torch.nn.Parameter(torch.rand(1, )))
    return torch.nn.ParameterList(params)

def get_moe_param(in_features: int, out_features: int, groups: int, experts: int, expert_chunks: int, std: float
                  ) -> typing.List[torch.nn.Parameter]:
    if experts:
        experts = groups if experts < 0 else experts
        out = orthonormal([in_features // groups, out_features // groups], std).view(1, in_features // groups, -1)
        out = out.repeat(experts // expert_chunks * groups, 1, 1).detach()
        gate = [orthonormal([in_features, experts], 1)]
        return [torch.nn.Parameter(copy.deepcopy(out)) for _ in range(expert_chunks)] + gate
    return [torch.nn.Parameter(conv_weight(in_features, out_features, 1, groups, std))]


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.bottleneck_group = ctx.model.bottleneck_group
        self.norm_power = ctx.model.norm_power
        self.groups0 = ctx.model.input_groups
        self.groups2 = ctx.model.output_groups
        self.experts0 = ctx.model.experts_in_input
        self.experts2 = ctx.model.experts_in_output
        self.jitter_epsilon = ctx.model.moe_jitter_epsilon
        self.activation_std = ctx.model.activation_std
        self.num_features = ctx.model.features
        self.expert_chunks = ctx.model.expert_chunks
        self.pkm = ctx.model.pkm.use_pkm
        self.pkm_layers = ctx.model.pkm.pkm_layer_depths
        self.ff_factor = ctx.model.feed_forward_intermediate_factor
        self.input_dropout = ctx.model.pkm.input_dropout
        self.query_dropout = ctx.model.pkm.query_dropout
        self.value_dropout = ctx.model.pkm.value_dropout
        self.pkm_topk = ctx.model.pkm.topk
        self.pkm_num_keys = ctx.model.pkm.num_keys
        self.pkm_layer = False
        self.pkm_heads = ctx.model.pkm.heads
        self.pkm_dim_head = ctx.model.pkm.dim_head
        self.pkm_keys = None
        self.pkm_values = None # Will be initialized upon cell copy if layer_num in pkm_layers
        self.norm = None
        self.input_dropout = ctx.model.pkm.input_dropout
        self.query_dropout = ctx.model.pkm.query_dropout
        self.value_dropout = ctx.model.pkm.value_dropout
        self.model_noise = ctx.model.use_riemann_noise
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        # conv_weight params:
        #   in_features: int, out_features: int, kernel_size: int, groups: int, std: float
        self.w0 = torch.nn.ParameterList(get_moe_param(ctx.model.features, intermediate * 3, self.groups0,
                                                       self.experts0, self.expert_chunks, ctx.model.activation_std))
        self.w1 = conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
                              ctx.model.activation_std)
        if ctx.model.use_riemann_noise:
            self.r0 = get_riemann_noise_params(ctx.model.features)
            self.r1 = get_riemann_noise_params(ctx.model.features)
            self.zero_holder = torch.Tensor([0]).to(torch.device('cuda'))
        else:
            self.r0 = None
            self.r1 = None
            self.zero_holder = None
        self.w2 = torch.nn.ParameterList(get_moe_param(intermediate, ctx.model.features, self.groups2,
                                                       self.experts2, self.expert_chunks, 1))
        self.idx: int = 0
        self._input_cache = torch.zeros([])
        self._cumsum_cache = torch.zeros([])
        if ctx.model.feature_shuffle:
            self.register_buffer("feature_shuffle0", torch.argsort(torch.randn(ctx.model.features)).view(1, -1, 1))
            self.register_buffer("feature_shuffle2", torch.argsort(torch.randn(intermediate)).view(1, -1, 1))
        else:
            self.feature_shuffle0 = None
            self.feature_shuffle2 = None

    def layer_check(self, layer_num: int):
        # Method to modify variables according to depth
        self.layer_num = layer_num
        if self.pkm:
            if layer_num in self.pkm_layers:
                self.pkm_layer = True
                self.experts2 = 0
                dim_query = self.pkm_dim_head * self.pkm_heads
                intermediate = int(self.num_features * self.ff_factor)
                if dim_query % 2 != 0:
                    raise ValueError("Invalid PKM dim query. \"model.pkm.dim_head\" * \
                    \"model.pkm_heads\" must equal a number divisible by two.")
                self.w1 = conv_weight(intermediate, self.num_features, self.kernel_size,
                                      self.groups2, self.activation_std)
                self.w2 = torch.nn.Parameter(torch.normal(torch.zeros(dim_query, self.num_features),
                                                          torch.ones(dim_query, self.num_features)))
                # w2 == "keys"
                self.pkm_keys = torch.nn.Parameter(torch.zeros(self.pkm_heads,
                                                             self.pkm_num_keys, 2, self.pkm_dim_head // 2))
                self.pkm_values = torch.nn.EmbeddingBag(self.pkm_num_keys ** 2, self.num_features, mode='sum', sparse=True)
                # Use MaskedBatchNorm1D if using mask objective
                self.norm = torch.nn.BatchNorm1d(self.num_features)
                init_(self.pkm_keys)
                init_(self.pkm_values.weight)
                self.input_dropout = torch.nn.Dropout(self.input_dropout)
                self.query_dropout = torch.nn.Dropout(self.query_dropout)
                self.value_dropout = torch.nn.Dropout(self.value_dropout)

    def reset_cache(self):
        self._cumsum_cache = torch.zeros([])
        self._input_cache = torch.zeros([])
        self.idx = 0

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.training:
            div = self.divisor()
        elif self.caching:
            self.idx += inp.size(2)
            div = torch.LongTensor([self.idx]).to(inp.device)
        else:
            self.idx = inp.size(2)
            div = torch.arange(self.idx, device=inp.device).view(1, 1, -1) + 1
        self._input_cache, self._cumsum_cache, out = linear_attention(inp, div,
                                                                      self.w0, self.r0, self.feature_shuffle0, self.groups0,
                                                                      self.experts0,
                                                                      self.w1, self.r1,
                                                                      self.w2, self.zero_holder, self.feature_shuffle2, self.groups2,
                                                                      self.experts2, self._input_cache,
                                                                      self._cumsum_cache, self.bottleneck_group,
                                                                      self.training, self.caching, self.idx,
                                                                      self.norm_power, self.jitter_epsilon,
                                                                      self.pkm_layer, self.pkm_keys, self.pkm_values,
                                                                      self.input_dropout, self.query_dropout,
                                                                      self.value_dropout, self.pkm_topk, self.pkm_num_keys, self.pkm_heads,
                                                                      self.norm, self.model_noise
                                                                      )
        out = out * self.init_scale
        return out

    def momentum(self, init_scale: float, deep: bool, layer_num: int):
        out = copy.deepcopy(self) if deep else copy.copy(self)
        out.init_scale = init_scale
        out.layer_check(layer_num)
        return out
