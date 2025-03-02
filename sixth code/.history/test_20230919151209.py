import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import random
from tkinter import messagebox, Tk
import pyautogui
from PIL import ImageGrab
import cv2
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
# time.sleep(2)

# pyautogui.moveTo(80, 100, duration=1)
# screenshot = pyautogui.screenshot()


screenshot = ImageGrab.grab()
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
target_image = cv2.imread('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png')
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# res = pyautogui.locateOnScreen('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png')
# x, y = pyautogui.center(res)

# result = cv2.matchTemplate(screenshot, target_image, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8

# # 获取匹配结果的位置信息
# loc = np.where(result <= threshold)

# 进入功能页
x1, y1 = [1137, 613]
pyautogui.moveTo(x1, y1, duration=1)
time.sleep(2)
pyautogui.click(clicks=2)
time.sleep(1)

# 进入‘我要来’
x2, y2 = [794, 756]
pyautogui.moveTo(x2, y2, duration=1)
time.sleep(2)
pyautogui.mouseDown(button='left')
time.sleep(1)
pyautogui.mouseUp(button='left')

# 进入‘预约入口’
x3, y3 = [794, 546]
pyautogui.moveTo(x3, y3, duration=1)
time.sleep(2)
pyautogui.click()
time.sleep(1)