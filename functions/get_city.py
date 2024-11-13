import requests
import json
import datetime

api_key = open("api_key", 'r').read()



# print(api_key)
# http://api.airvisual.com/v2/nearest_station?key={{YOUR_API_KEY}}
# url = "https://api.airvisual.com/v2/nearest_city?key=%s" %api_key
# # url = "https://api.airvisual.com/v2/city?city=Accra&state=Greater Accra&country=Ghana&key=%s" %api_key


# headers = {}
# payload = {}

# try:
#     response = requests.request("GET", url, headers=headers, data=payload)
# except:
#     pass
# def beautify(obj):
#     text = json.dumps(obj, sort_keys=True, indent=4)
#     return text


def time_elapsed(time):
    a = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
    b = datetime.datetime.strptime(datetime.datetime.now().isoformat(), "%Y-%m-%dT%H:%M:%S.%f")
    c = b - a
    return c

c = datetime.datetime.now()

# Displays Time
current_time = c.strftime('%H:%M')



# def get_city_details(addr):
#     """
#     obj[0] = city
#     obj[1] = country
#     obj[2] = aqius
#     obj[3] = weather
#     obj[4] = passed_time
#     obj[5] = w_icon (weather Icon)
#     obj[6] = humidity
#     """
#     url = "https://api.airvisual.com/v2/nearest_city?key=%s" %(api_key)
#     # url = "https://api.airvisual.com/v2/city?city=Accra&state=Greater Accra&country=Ghana&key=%s" %api_key


#     headers = {"X-FORWARDED-FOR": addr}
#     # headers = {"X-FORWARDED-FOR": '10.220.46.167'}
#     # headers = {"X-FORWARDED-FOR": "27.102.113.160"}
#     payload = {}

#     try:
#         response = requests.request("GET", url, headers=headers, data=payload)
#         print(response.text)
#     except:
#         pass
#     obj = json.loads(json.dumps(response.json()))
#     city = obj['data']['city']
#     country = obj['data']['country']
#     aqius = obj['data']['current']['pollution']['aqius'] #AQI US
#     pm25conc = obj['data']['current']['pollution']['pm2']['conc']
#     weather = obj['data']['current']['weather']['tp']
#     w_icon = obj['data']['current']['weather']['ic']
#     humidity = obj['data']['current']['weather']['hu']

#     given_time = obj['data']['current']['pollution']['ts']
#     # passed_time = round(time_elapsed(given_time).seconds/1000)
#     passed_time = current_time
#     # aqicn = obj['data']['current']['pollution']['aqicn']   # AQI GLOBAL
#     return city, country, aqius, passed_time, weather, w_icon, humidity, pm25conc

def get_city_details(addr):
    """
    obj[0] = city
    obj[1] = country
    obj[2] = aqius
    obj[3] = weather
    obj[4] = passed_time
    obj[5] = w_icon (weather Icon)
    obj[6] = humidity
    """
    api_link = "https://device.iqair.com/v2/65c642bd75a422a6c34bfca2"


    # headers = {"X-FORWARDED-FOR": addr}
    # headers = {"X-FORWARDED-FOR": '10.220.46.167'}
    # headers = {"X-FORWARDED-FOR": "27.102.113.160"}
    payload = {}

    response = requests.request("GET", api_link)
    # print(response.text)
    obj = json.loads(json.dumps(response.json()))
    city = "Accra"
    country = "Ghana"
    aqius = obj['current']['pm25']['aqius'] #AQI US
    pm25conc = obj['current']['pm25']['conc']
    weather = obj['current']['tp']
    w_icon = 0
    humidity = obj['current']['hm']


#     # given_time = obj['data']['current']['pollution']['ts']
#     # passed_time = round(time_elapsed(given_time).seconds/1000)
    passed_time = current_time
#     # aqicn = obj['data']['current']['pollution']['aqicn']   # AQI GLOBAL
    return city, country, aqius, passed_time, weather, w_icon, humidity, pm25conc


def get_device_rec():
    """
    obj[0] = city
    obj[1] = country
    obj[2] = aqius
    obj[3] = weather
    obj[4] = passed_time
    obj[5] = w_icon (weather Icon)
    obj[6] = humidity
    """
    response = requests.get("https://device.iqair.com/v2/65f2c4d9a6935ec6dbe485cb")
    obj = json.loads(json.dumps(response.json()))
    city = "Accra"
    country = "Ghana"
    if 'code' in obj:
        with open("data.json", "r") as f:
            obj = json.load(f)
            aqius = obj['historical']['instant'][0]['pm25']['aqius'] #AQI US
            pm25conc = obj['historical']['instant'][0]['pm25']['conc']
            weather = obj['historical']['instant'][0]['tp']
            w_icon = 0
            humidity = obj['historical']['instant'][0]['hm']
            passed_time = obj['historical']['instant'][0]['ts']
            passed_time = datetime.datetime.strptime(passed_time, "%Y-%m-%dT%H:%M:%S.%fZ")
            passed_time = passed_time.strftime("%A, %d. %B %Y %I:%M %p")
    else:
        with open("data.json", "w") as f:
            json.dump(obj, f)
        aqius = obj['historical']['instant'][0]['pm25']['aqius'] #AQI US
        pm25conc = obj['historical']['instant'][0]['pm25']['conc'] 
        weather = obj['historical']['instant'][0]['tp']
        w_icon = 0
        humidity = obj['historical']['instant'][0]['hm']
#     # given_time = obj['data']['current']['pollution']['ts']
#     # passed_time = round(time_elapsed(given_time).seconds/1000)
        passed_time = obj['historical']['instant'][0]['ts']
        passed_time = datetime.datetime.strptime(passed_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        passed_time = passed_time.strftime("%a, %d %B %Y %I:%M %p")
#     # aqicn = obj['data']['current']['pollution']['aqicn']   # AQI GLOBAL
    return city, country, aqius, passed_time, weather, w_icon, humidity, pm25conc


# api_link = "https://device.iqair.com/v2/65c642bd75a422a6c34bfca2"


# # headers = {"X-FORWARDED-FOR": addr}
# # headers = {"X-FORWARDED-FOR": '10.220.46.167'}
# # headers = {"X-FORWARDED-FOR": "27.102.113.160"}
# payload = {}

# response = requests.request("GET", api_link)
# # print(response.text)
# obj = json.loads(json.dumps(response.json()))
# print(obj)

# def jprint(obj):
#     # create a formatted string of the Python JSON object
#     text = json.dumps(obj, sort_keys=True, indent=4)
#     print(text)

if __name__ == '__main__':
#     # jprint(response.json())
#     # print(get_city_details("27.102.113.160"))
    print(get_city_details("197.255.71.156"))










# with open("countries.json", 'w') as f:
#     f.write(text)