import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,  # 使用正确的模型类
    GenerationConfig,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, create_reference_model
import numpy as np
from accelerate.data_loader import DataLoaderShard as _OriginalDataLoaderShard
from desktop_env import *  # 导入新创建的桌面环境
# 导入处理视觉信息的工具

from qwen_vl_utils import process_vision_info
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"



# Monkey‐patch to drop unsupported 'in_order' kwarg
def _patched_init(self, dataset, *args, **kwargs):
    kwargs.pop("in_order", None)
    return _OriginalDataLoaderShard.__orig_init__(self, dataset, *args, **kwargs)
_OriginalDataLoaderShard.__orig_init__ = _OriginalDataLoaderShard.__init__
_OriginalDataLoaderShard.__init__ = _patched_init

# --- 配置区 ---
# 1. 环境设置
ENV_GOAL = "请在桌面上执行以下操作：打开记事本，输入'Hello World'，然后保存文件。"
ENV_MAX_STEPS = 50  # 每个episode的最大步数

# 2. PPO 配置
PPO_OUTPUT_DIR = "./outputs_qwen25_ppo_desktop"  # 更新输出目录名称
PPO_LEARNING_RATE = 5e-6  # RL 学习率通常较小
PPO_BATCH_SIZE = 1  # 由于环境交互通常是串行的，batch size 常设为 1

# 3. 模型配置
MODEL_NAME = "./models/Qwen2.5-VL-3B-Instruct"  # 更新为本地模型路径
VALUE_MODEL_NAME = "./models/Qwen2.5-VL-3B-Instruct"  # 更新为本地模型路径
# VALUE_MODEL_NAME = "/home/chuangzhi/zzp/DRLInference/models/distilbert-base-uncased"  # 用于 PPO 的价值模型
TORCH_DTYPE = torch.float16  # 使用半精度以节省显存
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {DEVICE}")

# 4. LoRA 配置
LORA_R = 16  # LoRA 秩
LORA_ALPHA = 32  # LoRA 缩放因子
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 5. 生成配置
GENERATE_MAX_NEW_TOKENS = 4096  # 模型生成动作序列的最大长度
GENERATE_DO_SAMPLE = True  # 使用采样增加多样性
GENERATE_TEMPERATURE = 0.7  # 控制采样随机性
GENERATE_TOP_P = 0.9  # Top-P 采样

# 6. 训练配置
NUM_EPOCHS = 10  # 训练的总轮数 (episodes)


# 动作示例
example = """对于每一个目标，你需要输出正确的动作，动作形式如下：
[
  {
    "goal": "点击屏幕上的按钮",
    "action": "click(100, 200)"
  },
  {
    "goal": "在文本框中输入内容",
    "action": "type('Hello World')"
  },
  {
    "goal": "滚动页面",
    "action": "scroll(100)"
  },
  {
    "goal": "使用快捷键复制内容",
    "action": "hotkey(['ctrl', 'c'])"
  }
]

请根据屏幕截图分析当前状态，并输出一个合适的动作来完成目标。
"""

# 1. 创建桌面环境
env = DesktopEnv(
    goal=ENV_GOAL + "\n\n" + example,
    max_steps=ENV_MAX_STEPS,
    render_mode="human"
)

# 2. PPO 配置
ppo_config = PPOConfig(
    output_dir=PPO_OUTPUT_DIR,
    learning_rate=PPO_LEARNING_RATE,
    batch_size=PPO_BATCH_SIZE,
)

# 3. 加载模型和处理器
model_name = MODEL_NAME  # 使用配置中定义的模型路径
device = DEVICE

# 创建自定义设备映射

# device_map = {
#     # 视觉部分
#     "visual.patch_embed": 1,
#     **{f"visual.blocks.{i}": 1 for i in range(0, 5)},     # GPU 1: blocks 0-4
#     **{f"visual.blocks.{i}": 2 for i in range(5, 10)},    # GPU 2: blocks 5-9
#     **{f"visual.blocks.{i}": 3 for i in range(10, 15)},   # GPU 3: blocks 10-14
#     **{f"visual.blocks.{i}": 4 for i in range(15, 20)},   # GPU 4: blocks 15-19
#     **{f"visual.blocks.{i}": 5 for i in range(20, 25)},   # GPU 5: blocks 20-24
#     **{f"visual.blocks.{i}": 6 for i in range(25, 32)},   # GPU 6: blocks 25-31
#     "visual.norm": 2,                                     # GPU 2: norm

#     # 文本部分
#     "model.embed_tokens": 1,
#     **{f"model.layers.{i}": 1 for i in range(0, 4)},      # GPU 1: layers 0-3
#     **{f"model.layers.{i}": 2 for i in range(4, 8)},      # GPU 2: layers 4-7
#     **{f"model.layers.{i}": 3 for i in range(8, 12)},     # GPU 3: layers 8-11
#     **{f"model.layers.{i}": 4 for i in range(12, 16)},    # GPU 4: layers 12-15
#     **{f"model.layers.{i}": 5 for i in range(16, 20)},    # GPU 5: layers 16-19
#     **{f"model.layers.{i}": 6 for i in range(20, 28)},    # GPU 6: layers 20-27
#     "model.norm": 5,                                      # GPU 5: norm
#     "model.rotary_emb": 4,                                # GPU 4: rotary_emb
#     "lm_head": 6,                                         # GPU 6: lm_head（原在 GPU 3）
#     "visual.merger": 5,                                   # GPU 5: merger
#     "visual.patch_embed.proj": 1
# }


# 使用AutoModelForVision2Seq加载Qwen2.5-VL模型，使用自定义设备映射
vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map='auto',  # 使用自定义设备映射替代"auto"
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
    local_files_only=True,
)

# 打印模型分布情况
print("模型分布情况:")
if hasattr(vlm, 'hf_device_map'):
    for name, device in vlm.hf_device_map.items():
        print(f"{name}: {device}")

# 4. 注入 LoRA adapter
save_directory = "./lora_qwen25_finetuned_model"
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
    local_files_only=True  # 确保从本地加载处理器
)

vlm.resize_token_embeddings(len(processor.tokenizer))

# 尝试加载已有的LoRA模型，如果失败则创建新的
try:
    policy_model = PeftModel.from_pretrained(
        vlm, 
        save_directory,
        torch_dtype=torch.float16
    )
    print("成功加载已有的LoRA模型")
except Exception as e:
    print(f"加载LoRA模型失败: {e}，创建新的LoRA模型")
    policy_model = get_peft_model(vlm, lora_cfg)

    
# policy_model = policy_model.to(device)  # 不需要显式移动到设备，因为device_map已经处理

print(f"使用 {torch.cuda.device_count()} 块GPU进行并行训练")


# 为Qwen2.5-VL模型设置生成配置
policy_model.generation_config = GenerationConfig(
    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0,
    max_new_tokens=GENERATE_MAX_NEW_TOKENS,
    do_sample=GENERATE_DO_SAMPLE,
    temperature=GENERATE_TEMPERATURE,
    top_p=GENERATE_TOP_P,
)
policy_model = policy_model.to(device)
# 5. 参考模型和奖励模型
ref_model = policy_model.base_model.model  # 获取原始模型主体

# 获取隐藏维度大小
hidden_dim = vlm.model.config.hidden_size   # 或者直接用 4096

# 创建 Value Head
value_head = ValueHead(hidden_dim).to(device)

custom_value_model = CustomValueModel(config=vlm.config, vlm=vlm, value_head=value_head).to(device)

# 0. 占位数据集
ppo_dataset = Dataset.from_dict({"query": [""]})

# 8. 实例化 PPOTrainer
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=processor.tokenizer,
    model=policy_model,
    ref_model=ref_model,
    reward_model=policy_model,
    train_dataset=ppo_dataset,
    value_model=custom_value_model
)

# 9. 训练循环: 收集 (query,response,reward) 元组
all_queries, all_responses, all_rewards = [], [], []

for ep in range(NUM_EPOCHS):
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"错误: 重置环境时出错: {e}")
        continue
    
    done = False
    current_step = 0
    episode_reward = 0.0
    
    # 使用正确的模型类加载Qwen2.5-VL模型
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=TORCH_DTYPE,
    device_map='auto',
    trust_remote_code=True,
    local_files_only=True,
    )

    # 构建提示和图像
    print(f"\n--- 第 {ep + 1} 轮, 步骤 {current_step + 1} ---")
    current_step += 1
    
    # 获取环境观察
    chat_msgs = obs.get("chat_messages", [])
    
    # 确保有截图数据
    if "screenshot" not in obs or not isinstance(obs["screenshot"], (torch.Tensor, list, tuple, np.ndarray)):
        print("错误: 观察数据中缺少有效的截图 'screenshot'。")
        break
        
    try:
        image = Image.fromarray(obs["screenshot"]).convert("RGB").resize((400, 300))
    except Exception as e:
        print(f"错误: 无法从 obs['screenshot'] 创建 PIL Image: {e}")
        break

    # --- 准备模型输入 ---
    # 构建符合Qwen2.5-VL格式的消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # 直接传递PIL图像
                },
                {
                    "type": "text", 
                    "text": f"请根据屏幕截图分析当前状态，并输出一个合适的动作来完成目标：{ENV_GOAL}"
                },
            ],
        }
    ]
    
    # 使用processor处理消息
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 处理视觉信息
    try:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"错误: 处理视觉信息失败: {e}")
        # 备用方案
        inputs = processor(
            text=f"请描述这张图片并决定下一步动作来完成目标：{ENV_GOAL}",
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成动作序列
    outputs = policy_model.generate(
        **inputs,
        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
        do_sample=GENERATE_DO_SAMPLE
    )
    
    # 提取生成的文本
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_length:]
    response = processor.tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]

    print(f"模型生成的响应: {response}")
    
    # --- 与环境交互 ---
    print("在环境中执行动作...")
    try:
        obs, reward, terminated, truncated, info = env.step(response)
        done = terminated or truncated  # 更新 done 状态
        episode_reward += reward  # 累加奖励
        print(f"步骤奖励: {reward}, 是否终止: {terminated}, 是否截断: {truncated}")
    except Exception as e:
        print(f"错误: 执行 env.step 时出错: {e}")
        done = True  # 出现错误时终止当前 episode

    print(f"第 {ep + 1} 轮结束，总奖励: {episode_reward}")
    
    # 收集训练数据
    all_queries.append(inputs["input_ids"].squeeze(0))
    all_responses.append(outputs.squeeze(0))
    all_rewards.append(episode_reward)
    
    # 将收集的数据转换为数据集格式
    ppo_dataset = Dataset.from_dict({
        "query": [t.cpu().tolist() for t in all_queries],
        "response": [t.cpu().tolist() for t in all_responses],
        "reward": all_rewards
    })
    trainer.train_dataset = ppo_dataset
    print(ppo_dataset[0])  # 查看第一条数据
    # 启动 PPO 训练
    stats = trainer.train()
    print(f"PPO 训练完成: {stats}")

    # 10. 保存 LoRA adapter 和 tokenizer
    policy_model.save_pretrained(save_directory)
    processor.tokenizer.save_pretrained(save_directory)

print("训练完成！")
