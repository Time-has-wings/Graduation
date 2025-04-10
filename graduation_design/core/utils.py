import paddle
import argparse
import numpy as np
import builtins
from paddlenlp.transformers import LlamaConfig
from paddle.distributed.fleet import HybridCommunicateGroup

original_print = builtins.print

def print(*args, prefix="[graduation]", use_color=True, **kwargs):
    if use_color:
        color = "\033[92m"  # Green
        reset_color = "\033[0m"
        original_print(f"{color}{prefix}", *args, reset_color, **kwargs)
    else:
        original_print(prefix, *args, **kwargs)
        
def printf(string):
    prefix_string = "[DEBUG]: "
    print(prefix_string + string)
    
def set_seed(seed):
    printf(f'Setting random seed to {seed}')
    paddle.seed(seed)
    np.random.seed(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Llama model")
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--global_batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--iter_upper_limit', type=int, default=25, help='Upper limit of iterations')
    parser.add_argument('--hidden_size', type=int, default=4096, help='Hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--rms_norm_eps', type=float, default=1e-6, help='RMS norm epsilon')
    parser.add_argument('--num_attention_heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--dp_degree', type=int, default=1, help='Data parallel degree')
    parser.add_argument('--mp_degree', type=int, default=1, help='Model parallel degree')
    parser.add_argument('--pp_degree', type=int, default=1, help='Pipeline parallel degree')
    parser.add_argument('--mixed_precision', type=str, default='fp32', help='Mixed precision')
    parser.add_argument('--micro_batch_size', type=int, default=1, help='Micro batch size')
    parser.add_argument('--acc_step', type=int, default=8, help='Accumulation step')
    
    return parser.parse_args()

def generate_llama_config(args, hcg:HybridCommunicateGroup):
    config = {}
    
    # generate from args directly
    config['vocab_size'] = args.vocab_size
    config['hidden_size'] = args.hidden_size
    config['num_hidden_layers'] = args.num_hidden_layers
    config['num_attention_heads'] = args.num_attention_heads
    config['seq_length'] = args.seq_length
    config['rms_norm_eps'] = args.rms_norm_eps
    
    # generate according to args
    config['max_position_embeddings'] = config['seq_length']
    config['intermediate_size'] = 11008 if args.mp_degree != 1 or args.pp_degree != 1 else args.hidden_size * 8 // 3
    
    # set default values
    # config['use_flash_attention'] = True # [note] 该函数的真实作用在哪？ 其实是没有实际作用的,不会开启attn
    
    # tensor parallel
    if args.mp_degree != 1:
        config['tensor_parallel_degree'] = args.mp_degree
        config['tensor_parallel_rank'] = hcg.get_model_parallel_rank()
        config['tensor_parallel_output'] = True

    # print config
    print(f'config dict: {config}')
    
    # generate LlamaConfig
    llama_config = LlamaConfig(use_flash_attention=True, **config)
    print(f"llama_config : {llama_config}")
    return llama_config
    
def print_model_and_size(model:paddle.nn.Layer):
    print(f"Model: {model}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    
