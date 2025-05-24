import time
from desktop_env import DesktopEnv

# 创建环境
env = DesktopEnv(
    goal="请点击屏幕中央，然后输入'Hello World'",
    max_steps=10,
    render_mode="human"
)

# 重置环境
obs, info = env.reset()
print("环境已重置，开始测试...")

# 测试一些简单的动作
actions = [
    "click(500, 500)",  # 点击屏幕中央
    "type('Hello World')",  # 输入文本
    "hotkey(['ctrl', 'a'])",  # 全选
    "hotkey(['ctrl', 'x'])",  # 复制
    "hotkey(['ctrl', 'v'])"   # 粘贴
]

for i, action in enumerate(actions):
    print(f"\n执行动作 {i+1}/{len(actions)}: {action}")
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"奖励: {reward}")
    print(f"是否终止: {terminated}")
    print(f"是否截断: {truncated}")
    
    # 等待一段时间以便观察
    time.sleep(2)
    
    if terminated or truncated:
        break

print("\n测试完成！")
env.close()