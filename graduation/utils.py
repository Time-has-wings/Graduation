import paddle
import argparse
import builtins
import numpy as np

original_print = builtins.print

def print(*args, prefix="[graduation]", use_color=False, **kwargs):
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
    return parser.parse_args()