# -*- coding: utf-8 -*-
import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    GenerationConfig,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig # 如果需要量化加载，可以考虑添加
)
import numpy as np
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, create_reference_model
import gymnasium as gym
import browsergym.core  # 注册 openended 任务
from accelerate.data_loader import DataLoaderShard as _OriginalDataLoaderShard
import warnings, re

# 忽略特定警告 (例如来自 TRL 或 Peft 的未来版本警告)
warnings.filterwarnings("ignore", category=FutureWarning)

# Monkey‐patch 解决 'in_order' 参数不兼容问题
# 注意：这可能在未来版本的 accelerate 中不再需要
def _patched_init(self, dataset, *args, **kwargs):
    kwargs.pop("in_order", None)
    return _OriginalDataLoaderShard.__orig_init__(self, dataset, *args, **kwargs)
_OriginalDataLoaderShard.__orig_init__ = _OriginalDataLoaderShard.__init__
_OriginalDataLoaderShard.__init__ = _patched_init

# --- 配置区 ---
# 1. 环境设置
ENV_START_URL = "https://www.google.com/" # 可以换成其他起始 URL
ENV_HEADLESS = False # 设置为 True 则在后台运行浏览器，False 则显示浏览器窗口

# 2. PPO 配置
PPO_OUTPUT_DIR = "./outputs_llava_ppo_browsergym"
PPO_LEARNING_RATE = 5e-6 # RL 学习率通常较小
PPO_BATCH_SIZE = 1 # 由于环境交互通常是串行的，batch size 常设为 1
# PPO_EPOCHS = 3 # PPO 内部更新轮数，可以根据需要调整

# 3. 模型配置
MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" # 要使用的 VLM
VALUE_MODEL_NAME = "distilbert-base-uncased" # 用于 PPO 的价值模型 (可以是简单的预训练模型)
TORCH_DTYPE = torch.float16 # 使用半精度以节省显存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {DEVICE}")

# 4. LoRA 配置
LORA_R = 16 # LoRA 秩，可以调整
LORA_ALPHA = 32 # LoRA 缩放因子
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对更多层应用 LoRA

# 5. 生成配置
GENERATE_MAX_NEW_TOKENS = 4096 # 模型生成动作序列的最大长度
GENERATE_DO_SAMPLE = True # 使用采样增加多样性
GENERATE_TEMPERATURE = 0.7 # 控制采样随机性
GENERATE_TOP_P = 0.9 # Top-P 采样

# 6. 训练配置
NUM_EPOCHS = 10 # 训练的总轮数 (episodes)
MAX_STEPS_PER_EPISODE = 50 # 每轮训练的最大步数，防止无限循环

# --- 代码实现 ---

# 1. 环境设置
print("初始化 BrowserGym 环境...")
env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": ENV_START_URL,
                 "goal" :  "你的任务是搜索微软必应主页地址，你只能输出json指令，除了用于控制浏览器的json指令以外什么都不要输出",},
    
    # wait_for_user_message=True, # 等待来自环境的指示 (如果任务需要)
    headless=ENV_HEADLESS,
    # viewport_size={"width": 1280, "height": 720} # 可以指定浏览器视口大小
)

def parse_actions(response: str):
    """尝试将模型输出的字符串解析为 JSON 动作列表"""
    # 尝试找到 JSON 块
    try:
        # 通常模型会用 ```json ... ``` 包裹代码
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有 ```json，尝试直接解析整个响应（可能包含非 JSON 文本）
            # 寻找第一个 '[' 或 '{' 开始解析
            first_bracket = -1
            first_curly = -1
            try:
                first_bracket = response.index('[')
            except ValueError:
                pass
            try:
                first_curly = response.index('{')
            except ValueError:
                pass

            if first_bracket != -1 and (first_curly == -1 or first_bracket < first_curly):
                json_str = response[first_bracket:]
            elif first_curly != -1 and (first_bracket == -1 or first_curly < first_bracket):
                json_str = response[first_curly:]
            else: # 如果找不到，返回空列表
                 print(f"警告: 无法在响应中找到 JSON 列表或对象起始符号: {response}")
                 return []

            # 尝试找到匹配的结束括号/花括号
            # 这是一个简化的方法，可能不完美
            balance = 0
            start_char = json_str[0]
            end_char = ']' if start_char == '[' else '}'
            end_index = -1
            for i, char in enumerate(json_str):
                if char == start_char:
                    balance += 1
                elif char == end_char:
                    balance -= 1
                    if balance == 0:
                        end_index = i + 1
                        break
            if end_index != -1:
                json_str = json_str[:end_index]
            else: # 如果找不到匹配的结束符，尝试直接解析（可能失败）
                print(f"警告: 无法找到匹配的 JSON 结束符，尝试解析部分字符串: {json_str}")


        print(f"尝试解析 JSON: {json_str}")
        parsed = json.loads(json_str)
        # 确保返回的是列表
        if isinstance(parsed, dict):
            return [parsed] # 如果是单个动作字典，包装成列表
        elif isinstance(parsed, list):
            return parsed
        else:
            print(f"警告: 解析结果不是列表或字典: {parsed}")
            return []
    except json.JSONDecodeError as e:
        print(f"警告: JSON 解析失败 - {e}. 响应: {response}")
        # 可以尝试更宽松的解析或返回特定错误动作
        return [] # 返回空列表表示无法解析动作
    except Exception as e:
        print(f"警告: 解析动作时发生未知错误 - {e}. 响应: {response}")
        return []


# 2. PPO 配置
print("配置 PPO...")
ppo_config = PPOConfig(
    output_dir=PPO_OUTPUT_DIR,
    learning_rate=PPO_LEARNING_RATE,
    batch_size=PPO_BATCH_SIZE,
    mini_batch_size=PPO_BATCH_SIZE, # 通常与 batch_size 相同 (如果 batch_size=1)
    gradient_accumulation_steps=1,
    # ppo_epochs=PPO_EPOCHS,
    seed=42, # 设置随机种子以保证可复现性
    # log_with="tensorboard", # 可以使用 wandb 或 tensorboard 记录日志
    # tracker_project_name="llava_ppo_browsergym",
    # optimize_device_cache=True,
)

# 3. 加载模型和处理器
print(f"加载模型: {MODEL_NAME}...")
# quantization_config = BitsAndBytesConfig(load_in_8bit=True) # 可选：使用 8bit 量化加载以节省更多显存

# 3.1 加载视觉语言模型 (VLM)
vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto", # 自动将模型分片到可用设备 (GPU/CPU)
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
    # quantization_config=quantization_config # 应用量化配置 (如果使用)
)

# 3.2 加载处理器 (包含图像处理器和文本 tokenizer)
# 显式设置 use_fast=False 以避免警告（如果模型保存时未使用 fast tokenizer）
# 但通常建议尽可能使用 fast tokenizer
print(f"加载处理器: {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False # 根据警告调整，或尝试 True
)

# 检查并设置 pad_token (对于开放式生成很重要)
if processor.tokenizer.pad_token is None:
    print("警告: Tokenizer 没有 pad_token，将其设置为 eos_token。")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    vlm.config.pad_token_id = vlm.config.eos_token_id

# 动态图像标记数 (从处理器配置中读取)
dynamic_image_tokens = getattr(processor, "num_image_tokens", 1) # 提供默认值以防万一
print(f"处理器报告的动态图像标记数: {dynamic_image_tokens}")


# 4. 注入 LoRA 适配器
print("注入 LoRA 适配器...")
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SEQ_2_SEQ_LM" # 对于序列到序列或条件生成任务
)
# 将 LoRA 应用到 VLM 上，得到策略模型
policy_model = get_peft_model(vlm, lora_cfg)
policy_model.print_trainable_parameters() # 打印可训练参数信息

# 从原始 VLM 复制生成配置给策略模型
policy_model.generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME,
    pad_token_id=processor.tokenizer.pad_token_id,
    max_new_tokens=GENERATE_MAX_NEW_TOKENS,
    do_sample=GENERATE_DO_SAMPLE,
    temperature=GENERATE_TEMPERATURE,
    top_p=GENERATE_TOP_P,
)


# 5. 创建 PPO 需要的参考模型 (Reference Model)
# 参考模型是策略模型的一个冻结副本，用于计算 KL 散度惩罚项
print("创建参考模型...")
ref_model = create_reference_model(policy_model)
reward_model = create_reference_model(policy_model)

# --- 注意：关于 Reward Model 和 Value Model ---
# 在 TRL 的 PPO 实现中：
# - `reward_model` (可选): 可以是一个独立的模型，用于根据 (query, response) 对计算奖励。
#    如果未提供，奖励需要从外部（例如环境）传入 `trainer.step()`。
# - `value_model` (可选): 用于估计状态值 (state value)。如果未提供，TRL 会使用
#    `ref_model` 或 `policy_model` 的一部分（取决于配置）来创建价值头 (value head)。
#
# 在这个 browsergym 场景中：
# - 奖励来自环境 (`env.step()` 返回的 `reward`)。我们不需要独立的 `reward_model`。
# - 使用一个独立的、更小的模型作为 `value_model` 可能更高效。

# 6. 加载价值模型 (Value Model)
print(f"加载价值模型: {VALUE_MODEL_NAME}...")
# 注意：价值模型的选择需要考虑其输出是否适合 PPO 的价值估计。
# DistilBERT SST-2 输出的是分类 logits (通常是 2 类)。
# PPO 需要一个标量值。我们需要修改它或使用一个更适合的模型，或者让 TRL 自己创建价值头。
# **为了简单起见，我们先让 TRL 使用 ref_model 创建价值头，而不是加载外部 value_model。**
# 如果您确实想使用外部价值模型，需要确保其输出和用法与 TRL 的 PPO 兼容。

value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map={"": DEVICE} # 将价值模型也放到指定设备
)
# value_tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_NAME) # 需要对应的 tokenizer
# # 可能需要添加 value head 或修改模型输出以适应 PPO

# 7. 准备一个占位数据集 (PPO Trainer 需要)
# 实际数据将在与环境交互时动态生成
print("创建占位数据集...")
ppo_dataset = Dataset.from_dict({"query": ["placeholder query"]}) # 内容不重要

# 8. 实例化 PPOTrainer
print("实例化 PPOTrainer...")
# 注意：移除了 reward_model 和 value_model 参数，让 TRL 处理
# 如果使用外部 value_model，需要传递 value_model=value_model 和 tokenizer=value_tokenizer
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=processor.tokenizer,
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    train_dataset=ppo_dataset,
    value_model=value_model
)

# 9. PPO 训练循环
print("开始 PPO 训练循环...")
for ep in range(NUM_EPOCHS):
    print(f"\n--- 开始第 {ep + 1} / {NUM_EPOCHS} 轮训练 ---")
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"错误: 环境重置失败: {e}")
        continue # 跳过当前 episode
    done = False
    truncated = False
    current_step = 0
    episode_reward = 0.0

    while not done and not truncated and current_step < MAX_STEPS_PER_EPISODE:
        print(f"\n--- 第 {ep + 1} 轮, 步骤 {current_step + 1} ---")
        # 获取环境观察
        chat_msgs = obs.get("chat_messages", [])
        # 确保有截图数据
        if "screenshot" not in obs or not isinstance(obs["screenshot"], (torch.Tensor, list, tuple, np.ndarray)):
            print("错误: 观察数据中缺少有效的截图 'screenshot'。")
            # 可能需要跳过这一步或终止 episode
            break
        try:
            image = Image.fromarray(obs["screenshot"]).convert("RGB").resize((300, 200)) # 转换为 RGB
        except Exception as e:
            print(f"错误: 无法从 obs['screenshot'] 创建 PIL Image: {e}")
            break


        # --- 准备模型输入 (核心修改处) ---
        prompt_text = ""
        if chat_msgs:
            # 格式化聊天记录，并将文本 <image> 标记添加到最后一条用户消息前
            formatted_msgs = []
            for i, msg in enumerate(chat_msgs):
                 # 确保 content 是字符串
                 content_str = str(msg.get('content', ''))
                 role = msg.get('role', 'user') # 默认角色为 user
                 # 在最后一条消息 (且是 user 消息) 前添加文本 <image> 标记
                 if i == len(chat_msgs) - 1 and role == 'user':
                     # *** 直接使用字面量 "<image>" 字符串 ***
                     formatted_msgs.append({
                         "role": role,
                         "content": f"<image>\n{content_str}"  # 使用字面量字符串 "<image>"
                     })
                 else:
                     formatted_msgs.append({"role": role, "content": content_str})

            # 应用聊天模板
            try:
                # apply_chat_template 会识别 "<image>" 并正确处理
                prompt_text = processor.tokenizer.apply_chat_template(
                    formatted_msgs,
                    tokenize=False,
                    add_generation_prompt=True # 对于生成任务，这通常是必需的
                )
            except Exception as e:
                print(f"错误: 应用聊天模板失败: {e}")
                print(f"原始消息: {chat_msgs}")
                print(f"格式化消息: {formatted_msgs}")
                # 在默认提示中也使用字面量 "<image>"
                prompt_text = f"<image>\nUSER: 根据图像采取行动。\nASSISTANT:" # 使用字面量 "<image>"

        else:
            # 如果没有聊天记录，提供一个默认提示
            # 在默认提示中也使用字面量 "<image>"
            prompt_text = f"<image>\nUSER: 描述图像并决定下一步动作。\nASSISTANT:" # 使用字面量 "<image>"

        print(f"构建的提示文本 (应包含 '<image>'):\n{prompt_text}")

        # 使用处理器准备最终输入 (这部分保持不变)
        # processor 会处理包含 "<image>" 的文本 和 实际的图像数据
        # 使用处理器准备最终输入
        try:
            # *** 强制设定一个明确的最大长度 ***
            # 查阅 Qwen2-0.5b 的文档确认其推荐的最大长度，这里先用 2048 作为示例
            MAX_INPUT_LENGTH = 4096

            print(f"原始提示文本长度: {len(prompt_text)}") # 调试信息
            print(f"Tokenizer model_max_length: {processor.tokenizer.model_max_length}") # 调试信息
            print(f"强制最大输入长度: {MAX_INPUT_LENGTH}") # 调试信息

            inputs = processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
                padding="longest", # Padding 到 batch 内最长，但会被 max_length 限制
                truncation=True,   # 确保启用截断
                max_length=MAX_INPUT_LENGTH # *** 显式设置最大长度 ***
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # 移动到设备

            # 再次检查处理后的长度
            print(f"处理后 Input IDs shape: {inputs['input_ids'].shape}")

        except Exception as e:
            print(f"错误: Processor 处理输入失败: {e}")
            print(f"提示文本长度: {len(prompt_text)}")
            break # 出错则跳出当前 episode 的循环



        # 检查 pixel_values 是否存在且类型正确
        if "pixel_values" not in inputs or inputs["pixel_values"].dtype != TORCH_DTYPE:
            if "pixel_values" in inputs:
                print(f"警告: pixel_values 类型不匹配 ({inputs['pixel_values'].dtype})，转换为 {TORCH_DTYPE}")
                inputs["pixel_values"] = inputs["pixel_values"].to(TORCH_DTYPE)
            else:
                print("错误: Processor 未能生成 pixel_values。")
                break


        # --- 生成动作序列 ---
        print("生成动作...")
        # 准备 generate 函数需要的参数
        # 注意: 不应该直接传递 processor 返回的整个 `inputs` 字典给 generate
        # 因为可能包含 generate 不接受的键 (例如，如果 processor 返回了 'image_sizes')
        generate_kwargs = {
            **inputs,
            # 使用 policy_model 上附加的 generation_config
            # "max_new_tokens": GENERATE_MAX_NEW_TOKENS,
            # "do_sample": GENERATE_DO_SAMPLE,
            # "temperature": GENERATE_TEMPERATURE,
            # "top_p": GENERATE_TOP_P,
            # "pad_token_id": processor.tokenizer.pad_token_id,
        }

        # try:
            # 使用 torch.no_grad() 进行推理，节省显存并加速
        with torch.no_grad():
                outputs = policy_model.generate(**generate_kwargs)
        # except Exception as e:
        #     print(f"错误: 模型生成失败: {e}")
        #     # 打印输入形状可能有助于调试
        #     print(f"  Input IDs shape: {generate_kwargs['input_ids'].shape}")
        #     print(f"  Attention Mask shape: {generate_kwargs['attention_mask'].shape}")
        #     print(f"  Pixel Values shape: {generate_kwargs['pixel_values'].shape}")
        #     break


        # --- 解码和解析动作 ---
        # 只解码新生成的 token 部分
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        response_text = processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True # 跳过特殊 token 如 <eos>
        )[0]
        print(f"模型原始响应: {response_text}")

        # actions = parse_actions(response_text)
        actions = response_text
        print(f"解析出的动作: {actions}")

        # 如果没有解析出有效动作，可以决定是发送空动作还是停止
        if not actions:
             print("警告: 未能从模型响应中解析出有效动作。发送空动作列表。")
             # actions = [] # 保持为空列表

        # --- 与环境交互 ---
        print("在环境中执行动作...")
        try:
            # new_obs, reward, terminated, truncated, info = env.step(actions)
            # 修正：browsergym 的 step 返回值顺序是 obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated # 更新 done 状态
            episode_reward += reward # 累加奖励
            print(f"步骤奖励: {reward}, 是否终止: {terminated}, 是否截断: {truncated}")
        except Exception as e:
            print(f"错误: 执行 env.step 时出错: {e}")
            # 可能是动作格式错误或环境内部问题
            done = True # 出现错误时终止当前 episode


        # --- PPO 更新 ---
        # 准备 PPO step 需要的张量
        # query 是模型的输入 token ID
        # response 是模型的完整输出 token ID (输入 + 生成部分)
        # reward 是这一步获得的标量奖励
        query_tensor = inputs["input_ids"].squeeze(0) # 移除 batch 维度 (因为 batch_size=1)
        response_tensor = outputs.squeeze(0)          # 移除 batch 维度
        reward_scalar = torch.tensor([reward], device=DEVICE, dtype=torch.float) # 奖励需要是包含单个值的张量列表

        current_step += 1

        # 确保张量不为空
    if query_tensor.numel() > 0 and response_tensor.numel() > 0:
        ppo_dataset = Dataset.from_dict({
            "query": [t.cpu().tolist() for t in query_tensor],
            "response": [t.cpu().tolist() for t in response_tensor],
            "reward": reward_scalar
        })
        print("执行 PPO 更新...")
        stats = trainer.train()
        print(f"  PPO 统计信息: {stats}")
        # try:
        #     stats = trainer.train()
        #     print(f"  PPO 统计信息: {stats}")
        # except Exception as e:
        #     print(f"错误: trainer.step 执行失败: {e}")
        #     # 打印张量形状可能有助于调试
        #     print(f"  Query Tensor Shape: {query_tensor.shape}")
        #     print(f"  Response Tensor Shape: {response_tensor.shape}")
        #     print(f"  Reward Tensor: {reward_scalar}")
    else:
        print("警告: Query 或 Response 张量为空，跳过 PPO 更新。")

    print(f"\n--- 第 {ep + 1} 轮结束 ---")
    print(f"总步数: {current_step}")
    print(f"总奖励: {episode_reward}")
    # 在每轮结束后或定期保存模型检查点
    if (ep + 1) % 5 == 0 or ep == NUM_EPOCHS - 1: # 每 5 轮或最后一轮保存
        print(f"保存 LoRA 适配器到: {PPO_OUTPUT_DIR}/checkpoint_{ep+1}")
        save_path = f"{PPO_OUTPUT_DIR}/checkpoint_{ep+1}"
        policy_model.save_pretrained(save_path)
        processor.tokenizer.save_pretrained(save_path)


# 关闭环境
print("关闭 BrowserGym 环境...")
env.close()

# 10. 最终保存 LoRA 适配器和 tokenizer
print(f"训练完成。最终保存 LoRA 适配器到: {PPO_OUTPUT_DIR}/final")
final_save_path = f"{PPO_OUTPUT_DIR}/final"
policy_model.save_pretrained(final_save_path)
processor.tokenizer.save_pretrained(final_save_path)

print("脚本执行完毕。")