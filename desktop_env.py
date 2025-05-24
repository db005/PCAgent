import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import json
import mss
import pyautogui
from PIL import Image
from keyboard import Keyboard
from mouse import Mouse

class DesktopEnv(gym.Env):
    """基于键盘和鼠标操作的桌面环境，用于强化学习训练"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, goal="执行桌面操作", max_steps=50, render_mode=None):
        super().__init__()
        
        # 初始化键盘和鼠标控制器
        self.keyboard = Keyboard()
        self.mouse = Mouse()
        
        # 环境参数
        self.goal = goal  # 任务目标描述
        self.max_steps = max_steps  # 每个episode的最大步数
        self.current_step = 0
        self.render_mode = render_mode
        
        # 动作和观察空间
        # 动作空间是离散的，我们使用字符串表示动作
        self.action_space = spaces.Text(max_length=1024)
        
        # 观察空间是屏幕截图（RGB图像）
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 主显示器
            self.screen_width, self.screen_height = monitor["width"], monitor["height"]
        
        self.observation_space = spaces.Dict({
            "screenshot": spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8),
            "chat_messages": spaces.Sequence(spaces.Dict({
                "role": spaces.Text(max_length=10),
                "message": spaces.Text(max_length=1024)
            })),
            "url": spaces.Text(max_length=1024),
            "dom_object": spaces.Text(max_length=10240)
        })
        
        # 历史动作和状态
        self.action_history = []
        self.chat_messages = []
        
    def _get_obs(self):
        """获取当前环境的观察"""
        # 截取屏幕
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": self.screen_width, "height": self.screen_height}
            screenshot = np.array(sct.grab(monitor))
            
        # 构建观察字典
        obs = {
            "screenshot": screenshot,
            "chat_messages": self.chat_messages,
            "url": "desktop://local",  # 桌面环境没有URL概念，使用占位符
            "dom_object": json.dumps({"type": "desktop", "elements": []})  # 简化的DOM对象
        }
        return obs
    
    def _get_info(self):
        """获取额外信息"""
        return {
            "steps": self.current_step,
            "action_history": self.action_history
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置步数和历史
        self.current_step = 0
        self.action_history = []
        
        # 初始化聊天消息
        self.chat_messages = [{
            "role": "user",
            "message": f"目标: {self.goal}。请根据屏幕截图执行适当的操作。"
        }]
        
        # 给系统一些时间准备
        time.sleep(1)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _parse_action(self, action_str):
        """解析动作字符串并执行相应操作"""
        try:
            # 尝试解析为JSON格式的动作
            try:
                action_data = json.loads(action_str)
                if isinstance(action_data, list):
                    # 处理动作列表
                    for action_item in action_data:
                        self._execute_single_action(action_item.get("action", ""))
                    return True
                elif isinstance(action_data, dict):
                    # 处理单个动作
                    self._execute_single_action(action_data.get("action", ""))
                    return True
            except json.JSONDecodeError:
                # 如果不是JSON格式，尝试直接执行
                return self._execute_single_action(action_str)
        except Exception as e:
            print(f"执行动作时出错: {e}")
            return False
    
    def _execute_single_action(self, action_str):
        """执行单个动作"""
        # 记录动作
        self.action_history.append(action_str)
        
        # 解析动作字符串
        if "click" in action_str:
            # 点击操作: click(x, y) 或 click('element_id')
            try:
                if "(" in action_str and ")" in action_str:
                    args = action_str.split("(")[1].split(")")[0].strip()
                    if "," in args:
                        # 坐标点击
                        x, y = map(float, args.split(","))
                        self.mouse.click(x=x, y=y)
                    else:
                        # 元素点击 (简化处理，实际应用中需要元素识别)
                        print(f"模拟点击元素: {args}")
                        # 这里可以添加元素识别和点击逻辑
                return True
            except Exception as e:
                print(f"点击操作失败: {e}")
                return False
                
        elif "type" in action_str or "fill" in action_str:
            # 输入操作: type('text') 或 fill('element_id', 'text')
            try:
                if "(" in action_str and ")" in action_str:
                    args = action_str.split("(")[1].split(")")[0].strip()
                    if "," in args:
                        # 元素输入
                        element_id, text = args.split(",", 1)
                        text = text.strip().strip("'").strip('"')
                        print(f"模拟在元素 {element_id} 中输入: {text}")
                        self.keyboard.type(text)
                    else:
                        # 直接输入
                        text = args.strip().strip("'").strip('"')
                        self.keyboard.type(text)
                return True
            except Exception as e:
                print(f"输入操作失败: {e}")
                return False
                
        elif "scroll" in action_str:
            # 滚动操作: scroll(amount)
            try:
                if "(" in action_str and ")" in action_str:
                    amount = int(action_str.split("(")[1].split(")")[0].strip())
                    self.mouse.scroll(amount)
                return True
            except Exception as e:
                print(f"滚动操作失败: {e}")
                return False
                
        elif "hotkey" in action_str:
            # 热键操作: hotkey(['ctrl', 'c'])
            try:
                if "[" in action_str and "]" in action_str:
                    keys_str = action_str.split("[")[1].split("]")[0].strip()
                    keys = [k.strip().strip("'").strip('"') for k in keys_str.split(",")]
                    self.keyboard.hotkey(keys)
                return True
            except Exception as e:
                print(f"热键操作失败: {e}")
                return False
        
        # 如果没有匹配任何已知动作
        print(f"未知动作: {action_str}")
        return False
    
    def _calculate_reward(self, action_success):
        """计算奖励值"""
        # 基础奖励
        if not action_success:
            return -0.5  # 动作执行失败的惩罚
        
        # 基本奖励 (成功执行动作)
        reward = 0.1
        
        # 这里可以添加更复杂的奖励计算逻辑
        # 例如，基于任务完成度、动作效率等
        
        return reward
    
    def step(self, action):
        """执行一步动作"""
        self.current_step += 1
        
        # 执行动作
        action_success = self._parse_action(action)
        
        # 计算奖励
        reward = self._calculate_reward(action_success)
        
        # 检查是否达到最大步数
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 更新聊天消息
        self.chat_messages.append({
            "role": "assistant",
            "message": action
        })
        self.chat_messages.append({
            "role": "user",
            "message": "请继续执行下一步操作。"
        })
        
        # 获取新的观察和信息
        observation = self._get_obs()
        info = self._get_info()
        
        # 给系统一些时间执行动作
        time.sleep(0.5)
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # 在人类模式下，我们不需要额外渲染，因为动作直接在屏幕上执行
            pass
    
    def close(self):
        """关闭环境"""
        pass


import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        value = self.fc2(x)
        return value.squeeze(-1)  # 返回形状: [batch_size]


class VisionLanguageValueModel(nn.Module):
    def __init__(self, vlm, value_head):
        super().__init__()
        self.vlm = vlm  # Qwen2.5-VL 视觉语言模型
        self.value_head = value_head

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        # 假设最后一层 [CLS] token 作为状态表示
        cls_state = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        value = self.value_head(cls_state)
        return value



from transformers import PreTrainedModel

class CustomValueModel(PreTrainedModel):
    base_model_prefix = "vlm"  # 关键：定义 base_model_prefix
    def __init__(self, config, vlm, value_head):
        super().__init__(config)
        self.vlm = vlm
        self.value_head = value_head

    def forward(self, input_ids, attention_mask, pixel_values):
        # 自动将输入移到和模型相同的设备上
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device)

        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        cls_state = outputs.last_hidden_state[:, 0, :]
        value = self.value_head(cls_state)
        return value