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
# res = pyautogui.locateAllOnScreen('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png')
# x, y = pyautogui.center(res)

screenshot = ImageGrab.grab()
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
target_image = cv2.imread('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png', cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(np.array(screenshot), target_image, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8

# 获取匹配结果的位置信息
loc = np.where(result)
# pyautogui.moveTo(2756, 1702, duration=5)
# print(loc[0][0])
print(screenshot.size)