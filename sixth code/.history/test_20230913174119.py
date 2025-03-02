import smtplib
from email.mime.text import MIMEText
from email.header import Header

smtp_server = 'smtp.qq.com'
smtp_port = 465
sender = 'Xiaoqiang Zhu <abc2611617@qq.com>'
password = 'ryzliozueqthbfcg'

smtp_obj = smtplib.SMTP_SSL(smtp_server)
smtp_obj.connect(smtp_server, 465)
smtp_obj.login(sender, password)

msg = MIMEText('邮件内容', 'plain', 'utf-8')
msg['From'] = Header('发件人')
msg['To'] = Header('收件人', 'utf-8')
msg['Subject'] = Header('邮件标题', 'utf-8')

receiver = 'xqzhu@bjtu.edu.cn'
smtp_obj.sendmail(sender, receiver, msg.as_string())
smtp_obj.quit()