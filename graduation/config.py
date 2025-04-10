from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM
from paddle.distributed import fleet
import paddle.distributed as dist

def config_init(args):
    rank = dist.get_rank()
    
    vocab_size = args.vocab_size
    seq_length = args.seq_length
    hidden_size = args.hidden_size
    num_hidden_layers = args.num_hidden_layers
    rms_norm_eps = args.rms_norm_eps
    num_attention_heads = args.num_attention_heads
    
    max_position_embeddings = seq_length  # [note] 貌似是暂且这样设定的
    if args.mp_degree != 1 or args.pp_degree != 1:
        intermediate_size = 11008 # [note] 暂且这样设置
    else: 
        intermediate_size = hidden_size * 8 // 3  # [note]暂且这样设置
    # num_key_value_heads = num_attention_heads # 暂且不设置num_key_value_heads
    
    config = LlamaConfig(vocab_size=vocab_size,
                       hidden_size=hidden_size,
                       num_hidden_layers=num_hidden_layers,
                       num_attention_heads=num_attention_heads,
                       intermediate_size=intermediate_size,
                       max_position_embeddings=max_position_embeddings,
                       rms_norm_eps=rms_norm_eps,
                       seq_length=seq_length,
                       tensor_parallel_degree=args.mp_degree,
                       tensor_parallel_rank=rank,
                       tensor_parallel_output=True,
                       use_flash_attention=True,)
    print(f'>>>> config is \n {config}')
    return config

def arg_to_config(args) -> LlamaConfig:
    pass

def model_init(config):
    model = LlamaForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters()).item()
    param_bytes = total_params * 4  # float32 = 4 bytes
    param_mb = param_bytes / (1024 ** 2)  # MB
    param_gb = param_bytes / (1024 ** 3)  # GB
    print(f'>>> origin model is\n {model}')
    print(f">>> Total parameters: {total_params:,} (约 {total_params / 1e9:.2f} 亿)")
    print(f">>> Estimated memory usage: {param_mb:.2f} MB (约 {param_gb:.2f} GB)")
    return model

def model_fleet_init(model):
    model = fleet.distributed_model(model)
    total_params = sum(p.numel() for p in model.parameters()).item()
    param_bytes = total_params * 4  # float32 = 4 bytes
    param_mb = param_bytes / (1024 ** 2)  # MB
    param_gb = param_bytes / (1024 ** 3)  # GB
    print(f'>>> parallel model is\n {model}')
    print(f">>> Total parameters: {total_params:,} (约 {total_params / 1e9:.2f} 亿)")
    print(f">>> Estimated memory usage: {param_mb:.2f} MB (约 {param_gb:.2f} GB)")
    return model

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters()).item()
    param_bytes = total_params * 4  # float32 = 4 bytes
    param_mb = param_bytes / (1024 ** 2)  # MB
    param_gb = param_bytes / (1024 ** 3)  # GB1
    for p in model.parameters():
        print("p.dtype", p.dtype)
    print(f">>> Total parameters: {total_params:,} (约 {total_params / 1e9:.2f} 亿)")
    print(f">>> Estimated memory usage: {param_mb:.2f} MB (约 {param_gb:.2f} GB)")
    