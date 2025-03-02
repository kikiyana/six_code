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
screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)

res = pyautogui.locateOnScreen('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png')
x, y = pyautogui.center(res)

target_image = cv2.imread('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png', cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(np.array(screenshot), target_image, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

x, y = screenshot.shape

# 获取匹配结果的位置信息
loc = np.where(result <= threshold)
# pyautogui.moveTo(1100, 640, duration=5)
# print(loc[0][0])
print(loc)