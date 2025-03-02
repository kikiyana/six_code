import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import random
from tkinter import messagebox, Tk
import pyautogui


pyautogui.hotkey('command', 'space')

pyautogui.hotkey('command', 'f')
time.sleep(10)
pyautogui.write('国家博物馆')
time.sleep(10)
pyautogui.press('enter')
time.sleep(10)