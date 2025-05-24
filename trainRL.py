import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, create_reference_model
import gymnasium as gym
import browsergym.core  # register openended task
from accelerate.data_loader import DataLoaderShard as _OriginalDataLoaderShard

# ── Monkey‐patch to remove unsupported 'in_order' kwarg ─────────────────────
def _patched_init(self, dataset, *args, **kwargs):
    kwargs.pop("in_order", None)
    return _OriginalDataLoaderShard.__orig_init__(self, dataset, *args, **kwargs)

_OriginalDataLoaderShard.__orig_init__ = _OriginalDataLoaderShard.__init__
_OriginalDataLoaderShard.__init__ = _patched_init

# ── 1. Environment setup ────────────────────────────────────────────────────
env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": "https://www.xiaohongshu.com/"},
    wait_for_user_message=True,
    headless=False
)

def parse_actions(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return []

# ── 2. PPO configuration ───────────────────────────────────────────────────
ppo_config = PPOConfig(
    output_dir="./outputs",
    learning_rate=5e-6,
    batch_size=1,
    # ppo_epochs=3,
)

# ── 3. Load model & processor with static image‐token alignment ───────────────
model_name = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Static**: set num_image_tokens to match feature count (7269)
processor = AutoProcessor.from_pretrained(
    model_name,
    num_image_tokens=7269,
    trust_remote_code=True
)  # ensures text tokens == vision features :contentReference[oaicite:1]{index=1}

vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
vlm.to(device)

# ── 4. Inject LoRA adapter ───────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
vlm = get_peft_model(vlm, lora_cfg)
vlm.generation_config = GenerationConfig.from_pretrained(model_name)

# ── 5. Reference model for KL penalty ───────────────────────────────────────
ref_vlm = create_reference_model(vlm)

# ── 6. Placeholder dataset ───────────────────────────────────────────────────
ppo_dataset = Dataset.from_dict({"query": [""]})

# ── 7. Instantiate PPOTrainer ───────────────────────────────────────────────
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=processor.tokenizer,
    model=vlm,
    ref_model=ref_vlm,
    reward_model=None,
    train_dataset=ppo_dataset
)

# ── 8. Training loop ────────────────────────────────────────────────────────
for episode in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        # Extract prompt and screenshot
        chat_msgs = obs.get("chat_messages", [])
        prompt = "\n".join(f"{m['role']}: {m['message']}" for m in chat_msgs)
        image = Image.fromarray(obs["screenshot"])

        # Tokenize text+image with matching image‐token count
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Generate and step
        outputs = vlm.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=True
        )
        response = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        actions = parse_actions(response)

        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        # PPO update
        trainer.step(
            query=inputs["input_ids"],
            response=outputs,
            rewards=[reward]
        )
    env.close()

# ── 9. Save LoRA adapter ────────────────────────────────────────────────────
vlm.save_pretrained("lora_multimodal_adapter")
processor.tokenizer.save_pretrained("lora_multimodal_adapter")
