import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import random
from tkinter import messagebox, Tk
import pyautogui

pyautogui.moveTo(30, 30, duration=1)
pyautogui.hotkey('command', 'space')
time.sleep(1)
pyautogui.write('wechat')
pyautogui.press('enter')
time.sleep(10)


pyautogui.hotkey('command', 'f')
time.sleep(1)
pyautogui.write('国家博物馆')
time.sleep(1)
pyautogui.press('enter')
time.sleep(2)