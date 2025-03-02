import time
import pyautogui
import numpy as np

# pyautogui.moveTo(30, 30, duration=1)
# pyautogui.hotkey('command', 'space')
# time.sleep(1)
# pyautogui.write('wechat')
# time.sleep(1)
# pyautogui.press('enter')
# time.sleep(1)

# pyautogui.hotkey('command', 'f')
# time.sleep(1)
# pyautogui.write('guojiabowuguan')
# time.sleep(1)
# pyautogui.press('enter')
# time.sleep(1)


# # 进入功能页
# x1, y1 = [1137, 613]
# pyautogui.moveTo(x1, y1, duration=1)
# pyautogui.click(clicks=2)
# time.sleep(1)

# # 进入‘我要来’
# x2, y2 = [794, 756]
# pyautogui.moveTo(x2, y2, duration=1)
# pyautogui.mouseDown(button='left')
# time.sleep(1)
# pyautogui.mouseUp(button='left')

# # 进入‘预约入口’
# x3, y3 = [794, 546]
# pyautogui.moveTo(x3, y3, duration=1)
# pyautogui.click()
# time.sleep(1)

# # 点击“参观预约”
# x4, y4 = [744, 746]
# pyautogui.moveTo(x4, y4, duration=1)
# pyautogui.click()
# time.sleep(1)

# 点击同意协议
pyautogui.moveTo(744, 616, duration=1)
pyautogui.click()

pyautogui.scroll(-20)