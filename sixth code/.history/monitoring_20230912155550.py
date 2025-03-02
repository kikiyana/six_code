import requests
from bs4 import BeautifulSoup
import time

google_scholar = 'https://scholar.google.com.hk/citations?user=2fHc1o4AAAAJ&hl=zh-CN'

previous_citations = 0

def send_notification(new_citations):
    import smtplib