import requests
from bs4 import BeautifulSoup
import time

google_scholar = "https://scholar.google.com/citations?user=2fHc1o4AAAAJ&hl=zh-CN"

previous_citations = 0

def send_notification(new_citations):
    import smtplib
    from email.mime.text import MIMEText
    sender_email = 'abc2611617@gmail.com'
    sender_password = 'CC39fde21c'
    recipient_email = 'abc261167@qq.com'

    subject = 'You have new citations'
    message = f'Your citations have be updated by: {new_citations}'

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)    #mail.bjtu.edu.cn  465
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print('message have been sent')
    except Exception as e:
        print('error')


while True:
    
        response = requests.get(google_scholar)
        soup = BeautifulSoup(response.text, 'html.parser')

        # citations = soup.find('td', {'class': 'gsc_rsb_std'}).text.strip()
        indexes = soup.find_all("tr", "gsc_a_tr")
        print(soup)
        h_index = indexes[2].string
        i10_index = indexes[4].string
        citations = indexes[0].string
        print()
        citations = int(citations.replace(',', ''))

        if citations != previous_citations:
            print('citations have been changes：{previous_citations} -> {citations}')
            send_notification(citations)
            previous_citations  = citations
        
        time.sleep(100) # 3600*24
    # except Exception as e:
    #     print('错误')
