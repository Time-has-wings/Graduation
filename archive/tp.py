import paddle.distributed
from graduation.dummy_dataset import DummyDataset
from graduation.utils import set_seed, parse_args
from paddle.io import DistributedBatchSampler, DataLoader
from graduation.config import config_init, model_init, print_model_size
from paddle.distributed import fleet
import paddle
from paddle.profiler import Profiler, ProfilerTarget
from datetime import datetime
from paddlenlp.transformers import LlamaForCausalLM

def train(args):
    current_time = datetime.now()
    current_time = current_time.strftime("%Y%m%d-%H%M%S")
    
    # init strategy
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree
    }
    strategy.tensor_parallel_configs = {
        "tensor_init_seed": 1234 # 确保tp的初始化一致
    }
    strategy.amp = True
    strategy.amp_configs = {
        "init_loss_scaling": 32768.0,
        "use_pure_bf16": True,  # Enable pure bf16 training
    }
    
    # init fleet
    fleet.init(is_collective=True, strategy=strategy)
    config = config_init(args)
    model = LlamaForCausalLM(config)
    model = fleet.distributed_model(model)
    print('after fleet.distributed_model')
    print(model)
    print_model_size(model)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-4, weight_decay=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)    
    
    # Create AMP scaler
    scaler = paddle.amp.GradScaler(
        init_loss_scaling=32768.0,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=2,
    )
    
    # set dataloader
    local_batch_size = args.global_batch_size # [note] 暂且这样设置一下，后续再调整
    dataset = DummyDataset(vocab_size=args.vocab_size, seq_length=args.seq_length)
    sampler = DistributedBatchSampler(dataset, batch_size=local_batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler) 
    
    # start training
    iter_times = []
    paddle_start_event = paddle.device.Event(enable_timing=True)
    paddle_end_event = paddle.device.Event(enable_timing=True)
    for epoch in range(args.num_epochs):
        iter_times.clear()
        with Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], scheduler=(1, 10)) as prof:
            for batch_idx, (input, label) in enumerate(dataloader):
                if batch_idx >= args.iter_upper_limit:
                    print("Early stop.")
                    break
                
                print(f"Batch {batch_idx}, Memory Allocated: {paddle.device.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved: {paddle.device.cuda.memory_reserved() / 1024**3:.2f} GB")

                paddle.device.synchronize()
                paddle_start_event.record()
                
                with paddle.amp.auto_cast(enable=True, dtype='bfloat16', level='O2' ):
                    outputs = model(input, labels=label)
                    loss = outputs[0]

                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
                
                paddle_end_event.record()
                paddle.device.synchronize()
                
                prof.step()
                
                iter_time = paddle_start_event.elapsed_time(paddle_end_event) / 1000.0  # 转换为秒
                iter_times.append(iter_time)
                
                print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item():.4f}, time {iter_time:.4f}")
        
                # 手动释放显存
                del input, label, loss
                paddle.device.cuda.empty_cache()
                
                paddle.distributed.barrier() 
            
            prof.export(f"./profile_files/paddle_trace_{paddle.distributed.get_rank()}_{current_time}.json", format="json")
        
        iter_times = iter_times[1:] if iter_times else [] # remove the first item
        total_time = sum(iter_times) if iter_times else 0 
        avg_time = total_time / len(iter_times) if iter_times else 0
        min_time = min(iter_times) if iter_times else 0
        max_time = max(iter_times) if iter_times else 0
        print(f"Epoch {epoch} summary: total time {total_time:.4f}, avg time {avg_time:.4f}, min time {min_time:.4f}, max time {max_time:.4f}")
        
    # finish training
    if paddle.distributed.get_rank() == 0:
        print("Training finished.")

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    train(args)