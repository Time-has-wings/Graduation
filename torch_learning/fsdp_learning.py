import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    """
    初始化分布式环境。
    """
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    清理分布式环境。
    """
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """
    一个简单的神经网络模型。
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_model(rank, world_size):
    """
    准备并封装模型为 FSDP 模型。
    """
    # 创建模型并将其移动到对应的 GPU
    model = SimpleModel().to(rank)

    # 使用 FSDP 包装模型
    # 注意：FSDP 会将模型的参数和优化器状态分片到不同的 GPU 上
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片策略
        # 可选的其他参数：
        # mixed_precision=None,  # 如果需要混合精度，可以设置为 MixedPrecision 对象
        # device_id=rank,        # 指定当前进程的设备 ID
    )

    print(f"Rank {rank}: Model is wrapped with FSDP.")
    return fsdp_model

def main(rank, world_size):
    """
    主函数，用于初始化分布式环境并封装模型。
    """
    setup(rank, world_size)

    # 准备 FSDP 模型
    fsdp_model = prepare_model(rank, world_size)

    # 打印模型参数的形状（仅用于验证）
    for name, param in fsdp_model.named_parameters():
        print(f"Rank {rank}: Parameter {name} has shape {param.shape}")

    cleanup()

if __name__ == "__main__":
    main()