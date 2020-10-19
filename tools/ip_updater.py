import requests
import urllib.request
import time

while True:
    external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    r = requests.get("https://ipv6.dynv6.com/api/update?ipv6="+str(external_ip)+"&token=nxxAQ1kYLNbcYR3QbgnDDG48rx4zq-&zone=headshoot.dns.army")
    time.sleep(60)