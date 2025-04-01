import paddle.distributed
import paddle.distributed
from dummy_dataset import DummyDataset
from paddle_utils import set_seed, parse_args
from paddle.io import DistributedBatchSampler, DataLoader
from llama_config import config_init, model_init, print_model_size
from paddle.distributed import fleet
from paddle.optimizer import AdamW
import paddle
import time
from paddle.distributed.sharding import group_sharded_parallel

def train(args):
    # init fleet
    fleet.init(is_collective=True)
    
    # init sharding comminication group
    sharding_group = paddle.distributed.new_group(ranks=list(range(args.ngpu)))
    
    # set config and original model
    config = config_init(args)
    model = model_init(config)
    
    # mixed_precision
    if args.mixed_precision == 'bf16':
        print("Mixed precision: bf16")
        for param in model.parameters():
            param.value()._to(dtype=paddle.bfloat16)
    elif args.mixed_precision == 'fp16':
        print("Mixed precision: fp16")
        for param in model.parameters():
            param.value()._to(dtype=paddle.float16)
    else:
        print("Mixed precision not supported.")
    
    
    # set parallel 
    optimizer = AdamW(learning_rate=1e-4,parameters=model.parameters(),weight_decay=0.01)
    scaler = paddle.amp.GradScaler()
    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level='os_g',  # ZeRO-2
        group=sharding_group,
        offload=False,  # optional: offload to CPU
        buffer_max_size=2**23,  # gradient buffer size (about 8M elements)
        scaler=scaler  # pass GradScaler to support BF16
    )
    print('after group_sharded_parallel, model size is')
    print_model_size(model)
    
    # set dataloader
    dataset = DummyDataset(vocab_size=args.vocab_size, seq_length=args.seq_length)
    local_batch_size = args.global_batch_size // args.ngpu
    sampler = DistributedBatchSampler(dataset, batch_size=local_batch_size, shuffle=True, drop_last=True)  # drop_last means drop the last batch if it is not full
    dataloader = DataLoader(dataset, batch_sampler=sampler)  
    
    # start training
    iter_times = []
    for epoch in range(args.num_epochs):
        iter_times.clear()
        for batch_idx, (input, label) in enumerate(dataloader):
            if batch_idx >= args.iter_upper_limit:
                print("Early stop.")
                break
            
            paddle.device.synchronize()
            start_time = time.perf_counter()
            
            with paddle.amp.auto_cast(enable=True, dtype="bfloat16", level="O2"):
                outputs = model(input, labels=label)
                loss = outputs[0]
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
            
            paddle.device.synchronize()
            end_time = time.perf_counter()
            iter_times.append(end_time - start_time)
            
            print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item():.4f}, time {end_time - start_time:.4f}")
        
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