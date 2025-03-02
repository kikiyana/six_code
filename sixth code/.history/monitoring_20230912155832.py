import requests
from bs4 import BeautifulSoup
import time

google_scholar = 'https://scholar.google.com.hk/citations?user=2fHc1o4AAAAJ&hl=zh-CN'

previous_citations = 0

def send_notification(new_citations):
    import smtplib
    from email.mime.text import MIMEText
    sender_email = 'xqzhu@bjtu.edu.cn'
    sender_password = 'CC39fde21c'
    recipient_email = 'abc261167@qq.com'

    subject = 'You have new citations'
    message = f':{new_citations}'