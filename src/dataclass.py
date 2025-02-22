import pathlib
import typing

import torch
import yaml


class DataClass:
    def serialize(self):
        return serialize(self)


def serialize(instance: typing.Union[DataClass, typing.Dict[str, typing.Any]]):
    if isinstance(instance, DataClass):
        attributes = {key: getattr(instance, key) for key in dir(instance)
                      if not key.startswith('_') and not key.endswith('_')}
        return serialize({key: value for key, value in attributes.items() if not isinstance(value, typing.Callable)})
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}

class PKM(DataClass):
    use_pkm: bool = False
    topk: int = 32
    input_dropout: float = 0.0
    query_dropout: float = 0.0
    value_dropout: float = 0.0
    pkm_layer_depths: typing.List[int] = [4, 7]
    heads: int = 4
    dim_head: int = 256
    num_keys: int = 128
    splits: int = 2

class Model(DataClass):
    weight_sharing: bool = False
    checkpoint_path: str = "checkpoint.torch"
    steps_per_checkpoint: int = 0  # 0 -> disabled
    print_on_init: bool = True
    features: int = 256
    momentumnet_beta: float = 0.99  # The higher this is, the more numerically stable. BUT also lower impact per layer
    depth: int = 64
    batch_size: int = 128
    sequence_length: int = 256
    activation_std: float = 0.5893595616022745  # std(relu(torch.randn((inf,)))) == 0.5893595616022745
    input_embedding_std: float = 1.
    position_embedding_std: float = 1.
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    conv_kernel_size: int = 7
    feature_shuffle: bool = False
    feed_forward_intermediate_factor: float = 2.
    norm_power: int = 2  # 1 = mean(abs(x)), 2 = std, ...
    bottleneck_group: int = 1  # not all group counts are possible. it has to be divide self.features without residual
    offloading: bool = False
    input_groups: int = 1
    output_groups: int = 1
    experts_in_input: int = 0  # 0 to disable MoE
    experts_in_output: int = 0
    use_riemann_noise: bool = False
    moe_jitter_epsilon: float = 0.02
    expert_chunks: int = 1  # Increase it if not all MoE parameters fit onto the GPU
    pkm: PKM = PKM()


class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 256  # 256 (Prefetch) * 8 (Long) * 2048 (GPT context) * 256 (High Batch) = 1GiB RAM


class WandB(DataClass):
    project: str = 'gpt'
    entity: str = 'homebrewnlp'
    model_log_type: typing.Optional[str] = all  # One of "gradients", "parameters", "all", or None
    log_frequency: int = 16  # log gradients and parameters every N batches


class Log(DataClass):
    loss_steps_per_print: int = 32  # 0 -> off
    wandb: WandB = WandB()
    sample_completion_stmt: str = ''
    sample_seq_len: int = 512
    sample_temp: float = .1


class Offload(DataClass):
    device: str = "cpu"
    pin_memory: bool = True


class Zero(DataClass):
    cpu_offload: bool = True
    contiguous_gradients: bool = False
    overlap_comm: bool = True
    offload_param: Offload = Offload()
    offload_optimizer: Offload = Offload()
    stage3_max_live_parameters: float = 1
    stage3_max_reuse_distance: float = 1
    stage3_prefetch_bucket_size: float = 1
    stage3_param_persistence_threshold: float = 1


class OneCycle(DataClass):
    cycle_min_lr: float = 3e-4  # Base learning rate used at the start and end of cycle.
    cycle_max_lr: float = 1e-3  # Learning rate used in the middle of the cycle. Can be smaller than cycle_min_lr
    decay_lr_rate: float = 1e-4  # Decay rate for learning rate.
    cycle_first_step_size: int = 2048  # Number of training iterations in the increasing half of a cycle.
    cycle_second_step_size: typing.Optional[int] = None  # steps in second phase. None -> cycle_first_step_size
    cycle_first_stair_count: int = 0  # Number of stairs in first phase. 0 means staircase disabled
    cycle_second_stair_count: typing.Optional[int] = None  # Number of stairs in second phase
    decay_step_size: int = 2  # Every how many steps to decay lr. 0 -> no decay
    cycle_momentum: bool = True  # Whether to cycle `momentum` inversely to learning rate.
    cycle_min_mom: float = 0.8  # Initial momentum which is the lower boundary in the cycle for each parameter group.
    cycle_max_mom: float = 0.9  # Upper momentum boundaries in the cycle for each parameter group.
    decay_mom_rate: float = 0  # Decay rate for momentum
    last_batch_iteration: int = -1  # The index of the last batch. This parameter is used when resuming a training job.


class AdaptiveGradientClipping(DataClass):
    gradient_clipping: float = 0.01
    zero_division_eps: float = 1e-6
    eps: float = 1e-3


class SharpnessAwareMinimization(DataClass):
    enabled: bool = True
    step_size: bool = 0.05
    adaptive: bool = True


class Optimizer(DataClass):
    type: str = "AdamW"
    gradient_accumulation_steps: int = 1
    one_cycle: OneCycle = OneCycle()
    beta2: float = 0.95  # beta1 is controlled by one_cycle
    eps: float = 1e-8
    weight_decay: float = 0.01
    zero: Zero = Zero()
    agc = AdaptiveGradientClipping()
    sharpness_aware_minimization: SharpnessAwareMinimization = SharpnessAwareMinimization()

    # Shampoo hyper-params
    diagonal_eps: float = 1e-6
    matrix_eps: float = 1e-12
    inverse_exponent_override: int = 0
    start_preconditioning_step: int = 16
    preconditioning_compute_steps: int = 1
    statistics_compute_steps: int = 1
    block_size: int = 128
    best_effort_shape_interpretation: bool = True
    graft_type: str = 'adagrad'  # 'Adagrad' or 'SGD'
    nesterov: bool = True
    no_preconditioning_for_layers_with_dim_gt: int = 8192


class Eval(DataClass):
    cache: bool = False


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name in dir(instance):
        if name.startswith("_") or name.endswith("_") or name not in config:
            continue
        attr = getattr(instance, name)
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


class Context(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 config_path: typing.Optional[pathlib.Path] = None):
        self.log = Log()
        self.optimizer = Optimizer()
        self.dataset = Dataset()
        self.model = Model()
        self.eval = Eval()
        self.wandb = WandB()

        if config_path is not None:
            config = yaml.safe_load(config_path.read_text())

        if config is not None:
            init_class(self, config)
