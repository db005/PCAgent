import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    GenerationConfig,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, create_reference_model
from desktop_env import *  # 导入新创建的桌面环境
import numpy as np
import browsergym.core  # register openended task
from accelerate.data_loader import DataLoaderShard as _OriginalDataLoaderShard

# Monkey‐patch to drop unsupported 'in_order' kwarg
def _patched_init(self, dataset, *args, **kwargs):
    kwargs.pop("in_order", None)
    return _OriginalDataLoaderShard.__orig_init__(self, dataset, *args, **kwargs)
_OriginalDataLoaderShard.__orig_init__ = _OriginalDataLoaderShard.__init__
_OriginalDataLoaderShard.__init__ = _patched_init

# --- 配置区 ---
# 1. 环境设置
ENV_GOAL = "请在桌面上执行以下操作：打开记事本，输入'Hello World'，然后保存文件。"
ENV_HEADLESS = False # 设置为 True 则在后台运行浏览器，False 则显示浏览器窗口
ENV_MAX_STEPS = 50  # 每个episode的最大步数
# 2. PPO 配置
PPO_OUTPUT_DIR = "./outputs_llava_ppo_browsergym"
PPO_LEARNING_RATE = 5e-6 # RL 学习率通常较小
PPO_BATCH_SIZE = 1 # 由于环境交互通常是串行的，batch size 常设为 1
# PPO_EPOCHS = 3 # PPO 内部更新轮数，可以根据需要调整

# 3. 模型配置
MODEL_NAME = "/home/chuangzhi/zzp/DRLInference/models/Qwen2.5-VL-3B-Instruct"  # 更新为本地模型路径
VALUE_MODEL_NAME = "/home/chuangzhi/zzp/DRLInference/models/Qwen2.5-VL-3B-Instruct"  # 更新为本地模型路径
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

example = """对于每一个goal，你需要输出正确的action，action形式如下：[
  {
    "goal": "在网页上搜索'Python教程'",
    "action": "search('Python教程')"
  },
  {
    "goal": "点击网页上的'登录'按钮",
    "action": "click('login_button')"
  },
  {
    "goal": "在搜索框中输入'机器学习'",
    "action": "fill('search_box', '机器学习')"
  },
  {
    "goal": "选择下拉菜单中的'中文'",
    "action": "select_option('language_select', '中文')"
  },
  {
    "goal": "滑动页面到底部",
    "action": "scroll(0, 1000)"
  },
  {
    "goal": "在表单中输入用户名和密码并提交",
    "action": "fill('username', 'myuser') fill('password', 'mypassword') click('submit_button')"
  },
  {
    "goal": "在图片上点击以放大",
    "action": "click('image')"
  },
  {
    "goal": "打开新标签页并访问'https://www.example.com'",
    "action": "open_url('https://www.example.com')"
  },
  {
    "goal": "在评论框中输入'很有用，谢谢！'",
    "action": "fill('comment_box', '很有用，谢谢！')"
  },
  {
    "goal": "点击'返回首页'链接",
    "action": "click('home_link')"
  }
]"""
# # 1. Environment setup
# env = gym.make(
#     "browsergym/openended",
#     task_kwargs={"start_url": "https://www.google.com/",
#                  "goal" :  "你的任务是搜索微软必应主页地址。" + example,},
#     wait_for_user_message=True,
#     headless=False
# )
# 1. 创建桌面环境
env = DesktopEnv(
    goal=ENV_GOAL + "\n\n" + example,
    max_steps=ENV_MAX_STEPS,
    render_mode="human"
)


# Helper to parse JSON actions
def parse_actions(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return []

# 2. PPO configuration
ppo_config = PPOConfig(
    output_dir="./outputs",
    learning_rate=5e-6,
    batch_size=1,
)

# 3. Load model and processor
model_name = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_dtype = torch.float16
# vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_name,
#     device_map="auto", # 自动将模型分片到可用设备 (GPU/CPU)
#     torch_dtype=TORCH_DTYPE,
#     trust_remote_code=True,
#     # quantization_config=quantization_config # 应用量化配置 (如果使用)
# )
vlm = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map='auto',  # 使用自定义设备映射替代"auto"
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True,
    local_files_only=True,
)

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
    local_files_only=True  # 确保从本地加载处理器

)

# 4. Inject LoRA adapter
#    save_directory 即你之前保存 adapter 的目录
save_directory = "./lora_finetuned_model"
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
policy_model = PeftModel.from_pretrained(
    vlm, 
    save_directory,
    torch_dtype=torch.float16
)
# policy_model = get_peft_model(vlm, lora_cfg)
policy_model.generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME,
    pad_token_id=processor.tokenizer.pad_token_id,
    max_new_tokens=GENERATE_MAX_NEW_TOKENS,
    do_sample=GENERATE_DO_SAMPLE,
    temperature=GENERATE_TEMPERATURE,
    top_p=GENERATE_TOP_P,)

# 5. Reference and reward models
# ref_model = create_reference_model(policy_model)
# reward_model = create_reference_model(policy_model)
hidden_dim = vlm.model.config.hidden_size   # 或者直接用 4096
ref_model = policy_model.base_model.model  # 获取原始模型主体
reward_model = ref_model
# 创建 Value Head
value_head = ValueHead(hidden_dim).to(device)

# 构建完整的 Value Model
value_model = VisionLanguageValueModel(vlm, value_head)

# 移动到指定设备（device_map 已由 vlm 处理）
value_model.to(device)

custom_value_model = CustomValueModel(config=vlm.config, vlm=vlm, value_head=value_head).to(device)

# 6. Value model
# value_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# value_model = AutoModelForSequenceClassification.from_pretrained(
#     value_model_name,
#     torch_dtype=torch_dtype,
#     device_map={"": device}
# )

# 7. Placeholder dataset
ppo_dataset = Dataset.from_dict({"query": [""]})

# 8. Instantiate PPOTrainer
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=processor.tokenizer,
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    train_dataset=ppo_dataset,
    value_model=custom_value_model
)

# 9. Training loop: collect (query,response,reward) tuples
all_queries, all_responses, all_rewards = [], [], []
for ep in range(10):
    try:
        obs, _ = env.reset()
    except Exception as e:
        print(f"错误: 重置环境时出错: {e}")
        continue
    done = False
    current_step = 0
    episode_reward = 0.0
    while not done:
        # Build prompt and image
        print(f"\n--- 第 {ep + 1} 轮, 步骤 {current_step + 1} ---")
        current_step += 1
        # 获取环境观察
        chat_msgs = obs.get("chat_messages", [])
        # 确保有截图数据
        if "screenshot" not in obs or not isinstance(obs["screenshot"], (torch.Tensor, list, tuple, np.ndarray)):
            print("错误: 观察数据中缺少有效的截图 'screenshot'。")
            # 可能需要跳过这一步或终止 episode
            break
        try:
            image = Image.fromarray(obs["screenshot"]).convert("RGB").resize((400, 300))
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
                 content_str = str(msg.get('message', ''))
                 role = msg.get('role', 'user') # 默认角色为 user
                 # 在最后一条消息 (且是 user 消息) 前添加文本 <image> 标记
                 if i == len(chat_msgs) - 1 and role == 'user':
                     # *** 直接使用字面量 "<image>" 字符串 ***
                     formatted_msgs.append({
                         "role": role,
                         "content": f"<image>\n{content_str}"+"当前url："+obs["url"]+"当前dom节点"+json.dumps(obs["dom_object"])   # 使用字面量字符串 "<image>"
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
        
        print(processor.tokenizer.tokenize(prompt_text))
        
        # Prepare processor inputs
        inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate action sequence
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True
        )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        response = processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        # actions = parse_actions(response_text)
        actions = response#_text
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

    all_queries.append(inputs["input_ids"].squeeze(0))
    all_responses.append(outputs.squeeze(0))
    all_rewards.append(reward)
    # Convert collected data into dataset format for PPOTrainer
    ppo_dataset = Dataset.from_dict({
        "input_ids": [t.cpu().tolist() for t in all_queries],
        "response": [t.cpu().tolist() for t in all_responses],
        "reward": all_rewards
    })
    # Launch PPO training end-to-end
    stats = trainer.train()
    print(f"PPO training finished: {stats}")

    # 10. Save LoRA adapter & tokenizer
    policy_model.save_pretrained("lora_multimodal_adapter")
    processor.tokenizer.save_pretrained("lora_multimodal_adapter")
print('训练完成!')