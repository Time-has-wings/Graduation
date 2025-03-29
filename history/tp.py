import paddle.distributed
from dummy_dataset import DummyDataset
from paddle_utils import set_seed, parse_args
from paddle.io import DistributedBatchSampler, DataLoader
from llama_config import config_init, model_init, print_model_size
from paddle.distributed import fleet
import paddle
import time

def train(args):
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
    
    # init fleet
    fleet.init(is_collective=True, strategy=strategy)
    
    # set config and original model
    config = config_init(args)
    model = model_init(config)
    
    model = fleet.distributed_model(model)
    print('after fleet.distributed_model')
    print(model)
    print_model_size(model)

    optimizer = paddle.optimizer.AdamW(learning_rate=1e-4, parameters=model.parameters(), weight_decay=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)    
    
    # set dataloader
    local_batch_size = 8 # [note] 暂且这样设置一下，后续再调整
    dataset = DummyDataset(vocab_size=args.vocab_size, seq_length=args.seq_length)
    sampler = DistributedBatchSampler(dataset, batch_size=local_batch_size, shuffle=True)
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
            
            outputs = model(input, labels=label)
            loss = outputs[0]

            # 反向传播
            loss.backward()
            optimizer.step()
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