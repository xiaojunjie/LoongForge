"""
Cosmos3 训练器单卡测试

验证:
1. 训练器初始化
2. 模型构建
3. 数据加载
4. 单步训练
"""

import sys
import os
sys.path.insert(0, '/mnt/cluster/xiaojunjie/code/LoongForge')

# 模拟 LoongForge args (避免依赖 utils)
class MockArgs:
    """模拟 LoongForge 命令行参数"""
    def __init__(self):
        self.model_name = "Cosmos3-Nano"
        self.model_type = "cosmos3"
        self.seq_length = 45056
        self.global_batch_size = 1
        self.num_workers = 0
        self.gradient_accumulation_steps = 1
        self.context_parallel_size = 1
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.sequence_parallel = False
        self.distributed_parallelism = "ddp"  # 单卡用 DDP 测试
        self.train_iters = 10
        self.log_interval = 1
        self.eval_interval = 100
        self.save_interval = 10
        self.seed = 42
        self.lr = 2e-5
        self.weight_decay = 0
        self.lr_warmup_steps = 2
        self.lora_enabled = True
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_target_modules = ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]
        self.ema_enabled = True
        self.ema_rate = 0.1
        self.data_path = "./data"
        self.load_checkpoint_path = None
        self.vae_path = None
        self.output_dir = "./outputs/test_cosmos3"
        self.use_meta_tensor = False

# 测试导入
print("=" * 60)
print("Testing Cosmos3 Trainer (Single GPU)")
print("=" * 60)

# 导入核心模块
from cosmos_framework.train.diffusion.cosmos3.config import Cosmos3Config, get_cosmos3_config
from cosmos_framework.train.diffusion.cosmos3.model import cosmos3_model_provider, Cosmos3ModelWrapper
from cosmos_framework.train.diffusion.cosmos3.data import cosmos3_dataloader_provider, Cosmos3Dataset
print("\n✓ All modules imported successfully")

# 测试配置
print("\n" + "=" * 60)
print("Test 1: Config Loading")
print("=" * 60)
config = get_cosmos3_config("Cosmos3-Nano")
print(f"  Model name: {config.model.model_name}")
print(f"  Seq length: {config.model.max_num_tokens_after_packing}")
print(f"  Batch size: {config.global_batch_size}")
print("✓ Config loaded")

# 测试数据集
print("\n" + "=" * 60)
print("Test 2: Dataset Creation")
print("=" * 60)
dataset = Cosmos3Dataset(
    data_path="./data",
    seq_length=45056,
    max_samples=5,
)
print(f"  Dataset size: {len(dataset)}")
print(f"  Sample keys: {dataset[0].keys()}")
print("✓ Dataset created")

# 测试数据加载器
print("\n" + "=" * 60)
print("Test 3: Dataloader Creation")
print("=" * 60)
train_dataloader, val_dataloader = cosmos3_dataloader_provider(MockArgs())
print(f"  Train loader: {len(train_dataloader)} batches")
print(f"  Val loader: {val_dataloader is not None}")
print("✓ Dataloaders created")

# 测试模型构建
print("\n" + "=" * 60)
print("Test 4: Model Building")
print("=" * 60)
args = MockArgs()
model_config, model = cosmos3_model_provider(args)
print(f"  Config type: {type(model_config).__name__}")
print(f"  Model type: {type(model).__name__}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("✓ Model built")

# 测试训练步骤
print("\n" + "=" * 60)
print("Test 5: Training Step")
print("=" * 60)
model.train()
for iteration, batch in enumerate(train_dataloader):
    if iteration >= 2:
        break
    
    # 模拟 batch
    batch["batch_size"] = 1
    
    # 训练步骤
    output, loss = model.training_step(batch, iteration)
    print(f"  Iteration {iteration}: loss={loss.item():.4f}")
    
print("✓ Training steps completed")

# 测试优化器初始化
print("\n" + "=" * 60)
print("Test 6: Optimizer Initialization")
print("=" * 60)
optimizer, scheduler = model.init_optimizer_scheduler(
    optimizer_config={"lr": 2e-5, "betas": [0.9, 0.95], "eps": 1e-6, "fused": True, "keys_to_select": None},
    scheduler_config={"warm_up_steps": 2},
)
print(f"  Optimizer: {type(optimizer).__name__}")
print(f"  Scheduler: {type(scheduler).__name__}")
print(f"  Optimizer param groups: {len(optimizer.param_groups)}")
print("✓ Optimizer initialized")

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)
