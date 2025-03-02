import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import random
from tkinter import messagebox, Tk


def PolledDoctor(url, browser, interval):
    while(1):
        browser.get(url)
        browser.implicitly_wait(10)
        names = browser.find_elements(By.CSS_SELECTOR, '.doctorname')
        status = browser.find_elements(By.CSS_SELECTOR, '.doctoryuyue.flex.flex-align-center.flex-pack-center')
        for i in range(len(names)):
            if names[i].text == 'xxx':
                index = i
                break
        names[i].click()
        browser.implicitly_wait(10)
        windows = browser.window_handles
        browser.switch_to.window(windows[-1])
        try:
            knowing = browser.find_element(By.CSS_SELECTOR, '.knowimg')
            knowing.click()
            windows = browser.window_handles
            browser.switch_to.window(windows[-1])
        except:
            print("No knowing blcok")
        order_status_list = browser.find_elements(By.CSS_SELECTOR, '.yuyue.fr.doctoryuyue')
        for i in range(len(order_status_list)):
            order_text = order_status_list[i].get_attribute('innerText')
            if order_text == '':
                print('Order Text Wrong!')
            elif order_text != '约满' and order_text != '未开':
                window = Tk()
                window.attributes("-topmost", 1)
                window.withdraw()
                messagebox.showinfo('提示', '开始约号儿啦!')
        time.sleep(interval)

if __name__ == '__main__':
    option = webdriver.ChromeOptions()
    # option.add_argument('headless')  # 设置option
    s = Service("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome.exec")
    browser = webdriver.Chrome(service=s, options=option)
    url = "https://www.baidu.com/"  # 用自己的url替代
    interval = random.randint(10, 20)
    # interval = 5
    PolledDoctor(url, browser, interval)
    browser.quit()
