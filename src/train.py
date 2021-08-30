import math
import time

import deepspeed
import numpy as np
import torch

from src import model
from src.dataclass import Context

torch._C._debug_set_autodiff_subgraph_inlining(False)  # Not sure
torch._C._set_graph_executor_optimize(True)
torch._C._set_backcompat_broadcast_warn(False)
torch._C._set_backcompat_keepdim_warn(False)
torch._C._set_cudnn_enabled(True)
torch._C._set_mkldnn_enabled(True)
torch._C._set_mkldnn_enabled(True)
torch._C._set_cudnn_benchmark(True)
torch._C._set_cudnn_deterministic(False)
torch._C._set_cudnn_allow_tf32(True)
torch._C._set_cublas_allow_tf32(True)
torch._C._jit_set_inline_everything_mode(True)

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_nvfuser_enabled(False)


def parameter_count(net):
    return sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters()))


def main(ctx: Context):
    dtype = torch.float16 if ctx.model.float16 else torch.float

    config = {"train_batch_size": ctx.model.batch_size * ctx.optimizer.gradient_accumulation_steps,
              "gradient_accumulation_steps": ctx.optimizer.gradient_accumulation_steps,
              "optimizer": {"type": ctx.optimizer.type,
                            "params": {"betas": [0.9, ctx.optimizer.beta2],
                                       "eps": ctx.optimizer.epsilon,
                                       "weight_decay": ctx.optimizer.weight_decay
                                       }
                            },
              "fp16": {"enabled": ctx.model.float16},
              "zero_optimization": {"stage": 3,
                                    "cpu_offload": ctx.optimizer.zero.cpu_offload,
                                    "contiguous_gradients": ctx.optimizer.zero.contiguous_gradients,
                                    "overlap_comm": ctx.optimizer.zero.overlap_comm,
                                    "offload_param": {"device": ctx.optimizer.zero.offload_param.device,
                                                      "pin_memory": ctx.optimizer.zero.offload_param.pin_memory},
                                    "offload_optimizer": {"device": ctx.optimizer.zero.offload_optimizer.device,
                                                          "pin_memory": ctx.optimizer.zero.offload_optimizer.pin_memory},
                                    "stage3_max_live_parameters": ctx.optimizer.zero.stage3_max_live_parameters,
                                    "stage3_max_reuse_distance": ctx.optimizer.zero.stage3_max_reuse_distance,
                                    "stage3_prefetch_bucket_size": ctx.optimizer.zero.stage3_prefetch_bucket_size,
                                    "stage3_param_persistence_threshold": ctx.optimizer.zero.stage3_param_persistence_threshold,
                                    },
              "activation_checkpointing": {"cpu_checkpointing": True, "contiguous_memory_optimization": True},
              "steps_per_print": ctx.log.deepspeed_steps_per_print,
              "wall_clock_breakdown": ctx.log.wall_clock_breakdown,
              "dump_state": ctx.log.dump_state,
              "scheduler": {"type": "OneCycle",
                            "params": {"cycle_min_lr": ctx.optimizer.one_cycle.cycle_min_lr,
                                       "cycle_max_lr": ctx.optimizer.one_cycle.cycle_max_lr,
                                       "decay_lr_rate": ctx.optimizer.one_cycle.decay_lr_rate,
                                       "cycle_first_step_size": ctx.optimizer.one_cycle.cycle_first_step_size,
                                       "cycle_second_step_size": ctx.optimizer.one_cycle.cycle_second_step_size,
                                       "cycle_first_stair_count": ctx.optimizer.one_cycle.cycle_first_stair_count,
                                       "cycle_second_stair_count": ctx.optimizer.one_cycle.cycle_second_stair_count,
                                       "decay_step_size": ctx.optimizer.one_cycle.decay_step_size,
                                       "cycle_momentum": ctx.optimizer.one_cycle.cycle_momentum,
                                       "cycle_min_mom": ctx.optimizer.one_cycle.cycle_min_mom,
                                       "cycle_max_mom": ctx.optimizer.one_cycle.cycle_max_mom,
                                       "decay_mom_rate": ctx.optimizer.one_cycle.decay_mom_rate,
                                       "last_batch_iteration": ctx.optimizer.one_cycle.last_batch_iteration
                                       }
                            }
              }

    mod = model.LinearAttention(ctx)
    mod = mod.to(dtype=dtype)
    print(mod)
    parameters = parameter_count(mod)
    base = int(math.log10(parameters) / 3)
    print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')

    tensor = torch.load('out.tensor')
    tensor = tensor.long()

    batch_index = torch.arange(0, ctx.model.batch_size * ctx.optimizer.gradient_accumulation_steps).view(-1, 1)
    item_index = torch.arange(0, ctx.model.sequence_length).view(1, -1)
    batch_index = batch_index + item_index

    length = tensor.size(0) // ctx.model.sequence_length - 1
    len_len = len(str(length))

    mean_loss = 0
    curr_loss = 0
    mod, opt, _, lr_scheduler = deepspeed.initialize(model=mod, config=config, model_parameters=mod.parameters())

    while True:
        start_time = time.time()
        for i in range(1, 1 + length):
            src = tensor[batch_index].to(ctx.model.device)
            tgt = tensor[batch_index + 1].to(ctx.model.device)
            lss = mod(src.to(ctx.model.device), tgt.to(ctx.model.device))
            mod.backward(lss)
            with torch.no_grad():
                mod.step()
                lr_scheduler.step()
                curr_loss += lss.detach()
                batch_index += ctx.model.sequence_length
                if i % ctx.log.loss_steps_per_print == 0:
                    mean_loss += curr_loss
                    print(f"[{i:{len_len}d}/{length}] Loss: {curr_loss.item() / ctx.log.loss_steps_per_print:7.4f} -",
                          f"Mean: {mean_loss.item() / i:7.4f} |",
                          f"LR: {opt.param_groups[0]['lr']:.6f}",
                          f"| Batch/s: {i / (time.time() - start_time):.3f}")
                    curr_loss = 0
