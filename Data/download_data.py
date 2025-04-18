import requests

MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/lsrs.py"
params = {
    'year1': '2023', 'month1': '1', 'day1': '1',
    'year2': '2023', 'month2': '12', 'day2': '31',
    'fmt': 'csv'
}

response = requests.get(MESONET_URL, params=params)
with open('data/raw/mesonet_lsrs.csv', 'wb') as f:
    f.write(response.content)
