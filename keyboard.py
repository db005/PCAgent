import platform

import pyautogui
import pyperclip

from typing import Union


class Keyboard:
    def __init__(self) -> None:
        """初始化键盘类，根据操作系统设置默认的粉饰键(非macOS使用ctrl,macOS使用command)
        
        """
        if platform.platform() != "Darwin":
            self.modifier_key = "ctrl"
        else:
            self.modifier_key = "command"

    def type(self, text: str, interval: Union[float, None] = None) -> None:
        """模拟键盘输入文本

        Args:
            text (str): 要输入的字符串
            interval (float, optional): 字符之间的输入延迟(秒),None则使用剪贴板粘贴方式
        """
        if interval:
            pyautogui.write(text, interval=interval)
        else:
            clipboard_history = pyperclip.paste()
            pyperclip.copy(text)
            self.hotkey([self.modifier_key, "v"])
            pyperclip.copy(clipboard_history)

    def press(self, keys: str | list[str], interval: float = 0.1) -> None:
        """模拟按下并释放一个键或多个键

        Args:
            keys (str or list): 单个键名(如"a")或键名列表(如["a", "b"])
            interval (float, optional): 键之间的延迟(秒)
        """
        pyautogui.press(keys, presses=1, interval=interval)

    def hotkey(self, keys: list[str], interval: float = 0.1) -> None:
        """模拟组合键(如 ctrl+c)

        Args:
            keys (list): 键名列表(如["ctrl", "c"])
            interval (float, optional): 键之间的延迟
        """
        pyautogui.hotkey(*keys, interval=interval)

    def down(self, key: str):
        """按下某个键不释放."""
        pyautogui.keyDown(key)

    def up(self, key: str):
        """释放某个键."""
        pyautogui.keyUp(key)
