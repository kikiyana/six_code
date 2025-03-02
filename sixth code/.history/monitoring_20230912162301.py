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
    message = f'Your citations have be updated by: {new_citations}'

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        server = smtplib.SMTP('mail.bjtu.edu.cn', 465)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print('message have been sent')
    except Exception as e:
        print('error')

while True:
    try:
        response = requests.get(google_scholar)
        soup = BeautifulSoup(response