import torch
from PIL import Image
import mss
import numpy as np
import time
import argparse
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    GenerationConfig
)
from peft import PeftModel
from keyboard import Keyboard
from mouse import Mouse

def parse_args():
    parser = argparse.ArgumentParser(description="桌面操作自动化推理")
    parser.add_argument("--model", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
                        help="基础模型名称")
    parser.add_argument("--lora", type=str, default="./lora_desktop_adapter", 
                        help="LoRA适配器路径")
    parser.add_argument("--goal", type=str, default="打开记事本并输入'Hello World'", 
                        help="任务目标")
    parser.add_argument("--max_steps", type=int, default=10, 
                        help="最大执行步数")
    parser.add_argument("--delay", type=float, default=3.0, 
                        help="每步之间的延迟(秒)")
    return parser.parse_args()

def get_screenshot():
    """获取屏幕截图"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主显示器
        screenshot = np.array(sct.grab(monitor))
    return screenshot

def parse_and_execute_action(action_str, keyboard, mouse):
    """解析并执行动作"""
    print(f"执行动作: {action_str}")
    
    try:
        # 尝试解析为JSON格式的动作
        import json
        try:
            action_data = json.loads(action_str)
            if isinstance(action_data, list):
                # 处理动作列表
                for action_item in action_data:
                    execute_single_action(action_item.get("action", ""), keyboard, mouse)
                return True
            elif isinstance(action_data, dict):
                # 处理单个动作
                execute_single_action(action_data.get("action", ""), keyboard, mouse)
                return True
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试直接执行
            return execute_single_action(action_str, keyboard, mouse)
    except Exception as e:
        print(f"执行动作时出错: {e}")
        return False

def execute_single_action(action_str, keyboard, mouse):
    """执行单个动作"""
    if "click" in action_str:
        # 点击操作
        try:
            if "(" in action_str and ")" in action_str:
                args = action_str.split("(")[1].split(")")[0].strip()
                if "," in args:
                    # 坐标点击
                    x, y = map(float, args.split(","))
                    mouse.click(x=x, y=y)
                else:
                    # 元素点击 (简化处理)
                    print(f"模拟点击元素: {args}")
            return True
        except Exception as e:
            print(f"点击操作失败: {e}")
            return False
            
    elif "type" in action_str or "fill" in action_str:
        # 输入操作
        try:
            if "(" in action_str and ")" in action_str:
                args = action_str.split("(")[1].split(")")[0].strip()
                if "," in args:
                    # 元素输入
                    element_id, text = args.split(",", 1)
                    text = text.strip().strip("'").strip('"')
                    print(f"模拟在元素 {element_id} 中输入: {text}")
                    keyboard.type(text)
                else:
                    # 直接输入
                    text = args.strip().strip("'").strip('"')
                    keyboard.type(text)
            return True
        except Exception as e:
            print(f"输入操作失败: {e}")
            return False
            
    elif "scroll" in action_str:
        # 滚动操作
        try:
            if "(" in action_str and ")" in action_str:
                amount = int(action_str.split("(")[1].split(")")[0].strip())
                mouse.scroll(amount)
            return True
        except Exception as e:
            print(f"滚动操作失败: {e}")
            return False
            
    elif "hotkey" in action_str:
        # 热键操作
        try:
            if "[" in action_str and "]" in action_str:
                keys_str = action_str.split("[")[1].split("]")[0].strip()
                keys = [k.strip().strip("'").strip('"') for k in keys_str.split(",")]
                keyboard.hotkey(keys)
            return True
        except Exception as e:
            print(f"热键操作失败: {e}")
            return False
    
    # 如果没有匹配任何已知动作
    print(f"未知动作: {action_str}")
    return False

def main():
    args = parse_args()
    
    # 初始化键盘和鼠标控制器
    keyboard = Keyboard()
    mouse = Mouse()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和处理器
    print("加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True
    )
    
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # 加载LoRA适配器
    try:
        model = PeftModel.from_pretrained(
            base_model, 
            args.lora,
            torch_dtype=torch.float16
        )
        print("成功加载LoRA适配器")
    except Exception as e:
        print(f"加载LoRA适配器失败: {e}，使用基础模型")
        model = base_model
    
    # 设置生成配置
    model.generation_config = GenerationConfig.from_pretrained(
        args.model,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    print(f"目标任务: {args.goal}")
    print(f"最大步数: {args.max_steps}")
    print(f"每步延迟: {args.delay}秒")
    print("准备开始执行...")
    time.sleep(3)  # 给用户一些准备时间
    
    # 执行循环
    for step in range(args.max_steps):
        print(f"\n--- 步骤 {step+1}/{args.max_steps} ---")
        
        # 获取屏幕截图
        screenshot = get_screenshot()
        image = Image.fromarray(screenshot).convert("RGB").resize((400, 300))
        
        # 准备提示
        prompt = f"<image>\nUSER: 目标: {args.goal}\n请根据屏幕截图分析当前状态，并输出一个合适的动作来完成目标。\nASSISTANT:"
        
        # 准备模型输入
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成动作
        print("生成动作中...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True
        )
        
        # 解码响应
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        response = processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        print(f"模型生成的响应: {response}")
        
        # 执行动作
        success = parse_and_execute_action(response, keyboard, mouse)
        
        # 等待指定的延迟时间
        print(f"等待 {args.delay} 秒...")
        time.sleep(args.delay)
        
        # 询问是否继续
        user_input = input("按Enter继续，输入'q'退出: ")
        if user_input.lower() == 'q':
            break
    
    print("\n执行完成！")

if __name__ == "__main__":
    main()