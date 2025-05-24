from datasets import load_dataset
import torch
from transformers import AutoTokenizer, LlavaOnevisionForConditionalGeneration
from accelerate.data_loader import DataLoaderShard as _OriginalDataLoaderShard

# Monkey‐patch to drop unsupported 'in_order' kwarg
def _patched_init(self, dataset, *args, **kwargs):
    kwargs.pop("in_order", None)
    return _OriginalDataLoaderShard.__orig_init__(self, dataset, *args, **kwargs)
_OriginalDataLoaderShard.__orig_init__ = _OriginalDataLoaderShard.__init__
_OriginalDataLoaderShard.__init__ = _patched_init

# Load the dataset
dataset = load_dataset("json", data_files="goal_action_pairs.json")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(text=examples['text'], text_target=examples['labels'], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

# Load the base model
base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    device_map="auto", # 自动将模型分片到可用设备 (GPU/CPU)
    torch_dtype=torch.float32,
    trust_remote_code=True,
    
    # quantization_config=quantization_config # 应用量化配置 (如果使用)
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.enable_input_require_grads()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)
model.config.gradient_checkpointing = False
trainer.train()

# Define the directory where you want to save the model
save_directory = "./lora_finetuned_model"

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)