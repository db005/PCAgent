import mss
import pyautogui

pyautogui.FAILSAFE = False # 禁用安全保护（防止鼠标触边中断程序）

w, h = pyautogui.size() # 获取屏幕逻辑分辨率（如1920x1080）
with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": w, "height": h}
    mss_image = sct.grab(monitor) # 截取全屏图像（实际像素分辨率）
    scaling_factor = int(mss_image.width / w) # 计算缩放因子（适配高DPI屏幕）


class Mouse:
    def scroll(self, clicks):
        """控制鼠标滚轮滚动.
        
        Args:
            clicks (int): 滚动的单位数，正值向上滚动，负值向下滚动.
        """
        pyautogui.scroll(clicks)

    def move(self, x: float | None = None, y: float | None = None):
        """将鼠标移动到指定的坐标(适配高DPI屏幕).
        
        Args:
            x (float, optional): 目标x坐标.
            y (float, optional): 目标y坐标.
        """
        if x is not None:
            x = x / scaling_factor
        if y is not None:
            y = y / scaling_factor
        pyautogui.moveTo(x, y)

    def click(
        self,
        x: float | None = None,
        y: float | None = None,
        button="left",
        clicks=1,
        interval=0.0,
    ):
        """在指定坐标执行鼠标点击操作.
        
        Args:
            x (float, optional): 目标x坐标.
            y (float, optional): 目标y坐标.
            button (str, optional): 鼠标按钮名称，默认为"left".
            clicks (int, optional): 点击次数，默认为1.
            interval (float, optional): 点击间隔时间，默认为0.0秒.
        """
        if x is not None:
            x = x / scaling_factor
        if y is not None:
            y = y / scaling_factor
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)

    def down(self):
        """按下鼠标按键（不释放）."""
        pyautogui.mouseDown()

    def up(self):
        """释放鼠标按键."""
        pyautogui.mouseUp()
