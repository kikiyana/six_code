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
# res = pyautogui.locateOnScreen('/Users/zhuxiaoqiang/Desktop/IEEE Trans/sixth code/button1.png')
x, y = pyautogui.center(res)
print(x, y)