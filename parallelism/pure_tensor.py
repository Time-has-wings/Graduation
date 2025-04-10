from graduation.dummy_dataset import DummyDataset
from graduation.utils import set_seed, parse_args, generate_llama_config, print_model_and_size, print_size
from paddle.io import DistributedBatchSampler, DataLoader
import nvtx
from ctypes import cdll
from paddle.distributed import fleet
from paddlenlp.transformers import LlamaForCausalLM
import paddle
import os

def train(args):
    # get gloal rank
    global_rank = paddle.distributed.get_rank()
    
    # init strategy, fleet and hcg
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree,
        "mp_configs": {
            "need_broadcast_data": False,
        }
    }
    strategy.tensor_parallel_configs = {
        "tensor_init_seed": 1234 
    }
    strategy.amp = True
    strategy.amp_configs = {
        "init_loss_scaling": 32768.0,
        "use_pure_bf16": True, 
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    
    # init llama_config and llama_model
    llama_config = generate_llama_config(args, hcg)    
    llama_model = LlamaForCausalLM(llama_config)
    llama_model = fleet.distributed_model(llama_model)
    print_size(llama_model)
    
    # init optimizer
    optimizer = paddle.optimizer.Adam(
        parameters=llama_model.parameters(),
        learning_rate=1e-4,
        weight_decay=0.01,
        multi_precision=True
    )
    optimizer = fleet.distributed_optimizer(optimizer)
    
    # create amp scaler
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=32768.0,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=2,
    )
    
    # create dataloader
    dataset = DummyDataset(vocab_size=args.vocab_size, seq_length=args.seq_length)
    sampler = DistributedBatchSampler(dataset, batch_size=args.global_batch_size, num_replicas=args.dp_degree, rank=hcg.get_data_parallel_rank(), shuffle=True)  # when pure tp, the batch_size is global batch_size
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    
    # start training
    iter_times = []
    paddle_start_event = paddle.device.Event(enable_timing=True)
    paddle_end_event = paddle.device.Event(enable_timing=True)
    libcudart = cdll.LoadLibrary(os.getenv("cudart_path"))
    
    for epoch in range(args.num_epochs):
        for step, (input, label) in enumerate(dataloader):
            if step > args.iter_upper_limit:
                break
            
            paddle.device.synchronize()
            
            if (step == 8):
                libcudart.cudaProfilerStart()
            
            with nvtx.annotate(f'paddle_rank{global_rank}_iter{step}'):
                paddle_start_event.record()
                
                with paddle.amp.auto_cast(enable=True, dtype="bfloat16", level="O2"):
                    outputs = llama_model(input, labels=label)
                    loss = outputs[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                paddle_end_event.record()
                paddle.device.synchronize()
            
            iter_times.append(paddle_start_event.elapsed_time(paddle_end_event) / 1e3)
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.numpy()}, Iter Time: {iter_times[-1]}s")
        
            paddle.distributed.barrier()
            if (step == 13):
                libcudart.cudaProfilerStop()
            
        iter_remove_second = iter_times[2:] if iter_times else []
        total_time = sum(iter_remove_second) if iter_remove_second else 0
        avg_time = total_time / len(iter_remove_second) if iter_remove_second else 0
        min_time = min(iter_remove_second) if iter_remove_second else 0
        max_time = max(iter_remove_second) if iter_remove_second else 0
        print(f"Epoch {epoch} summary: total time {total_time:.4f}, avg time {avg_time:.4f}, min time {min_time:.4f}, max time {max_time:.4f}")
        
    if global_rank == 0:
        print("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)