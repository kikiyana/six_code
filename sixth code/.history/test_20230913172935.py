import smtplib
from email.mime.text import MIMEText
from email.header import Header

smtp_server = 'mail.bjtu.edu.cn'
smtp_port = 465
sender = 'xqzhu@bjtu.edu.cn'
password = 'CC39fde21c'

smtp_obj = smtplib.SMTP_SSL(smtp_server)
smtp_obj.connect(smtp_server, 465)
smtp_obj.login(sender, password)

msg = MIMEText('邮件内容', 'plain', 'utf-8')
msg['From'] = Header('发件人', 'utf-8')
msg['To'] = Header('收件人', 'utf-8')
msg['Subject'] = Header('邮件标题', 'utf-8')

receiver = '729979478@qq.com'
smtp_obj.sendmail(sender, receiver, msg.as_string())
smtp_obj.quit()