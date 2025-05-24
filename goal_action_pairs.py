import json
import random

# 定义不同类型的动作和目标
goals = [
    "在网页上搜索'{keyword}'",
    "点击网页上的'{element}'按钮",
    "在搜索框中输入'{query}'",
    "选择下拉菜单中的'{option}'",
    "滑动页面到底部",
    "在表单中输入用户名和密码并提交",
    "在图片上点击以放大",
    "打开新标签页并访问'{url}'",
    "在评论框中输入'{comment}'",
    "点击'{element}'链接"
]

actions = [
    "search('{query}')",
    "click('{element}')",
    "fill('{element}', '{value}')",
    "select_option('{element}', '{option}')",
    "scroll(0, {distance})",
    "open_url('{url}')"
]

# 生成1000个目标与动作的组合
data = []
for _ in range(1000):
    goal = random.choice(goals).format(
        keyword=random.choice(['Python教程', '机器学习', '深度学习', 'Web开发']),
        element=random.choice(['login', 'submit', 'search', 'image', 'home']),
        query=random.choice(['深度学习', 'Python', 'AI', '机器学习']),
        option=random.choice(['中文', 'English', 'Spanish']),
        url=random.choice(['https://www.xiaohongshu.com', 'https://www.google.com', 'https://www.bilibili.com']),
        comment=random.choice(['很有用，谢谢！', '不太满意', '内容不错，但有些地方需要改进'])
    )
    action = random.choice(actions).format(
        query=random.choice(['深度学习', 'Python', 'AI', '机器学习']),
        element=random.choice(['login_button', 'search_box', 'submit_button', 'image']),
        value=random.choice(['myuser', 'mypassword']),
        option=random.choice(['中文', 'English']),
        distance=random.randint(100, 2000),
        url=random.choice(['https://www.xiaohongshu.com', 'https://www.google.com', 'https://www.bilibili.com'])
    )
    data.append({"text": goal, "labels": action})

# 保存为JSON文件
with open('goal_action_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("生成的goal_action_pairs.json已保存")
