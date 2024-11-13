from io import StringIO
import requests
import pandas as pd 
import json 

# apikey = "iEvliA2IW3H5Zo2eU0ojtUVpNDd13tXd" 

# api_url = "https://airapi.airly.eu/v2/measurements/nearest?lat=5.6502465&lng=-0.1832575&maxDistanceKM=5&maxResults=3"

# response = requests.get(api_url, headers={'apikey': apikey})

# df = 

# with open('airly_data.json', 'w') as f:
#     f.write(response.text)

# # print(response.text)
# with open('airly_data.json', 'r') as f:
#     data = json.load(f)
#     pm25conc = data['current']['values'][1]['value']
#     pm10conc = data['current']['values'][2]['value']
#     humidity = data['current']['values'][4]['value']
#     tempurature = data['current']['values'][5]['value']

# fin_data = pm25conc, pm10conc, humidity, tempurature
# print(fin_data)

apikey = "iEvliA2IW3H5Zo2eU0ojtUVpNDd13tXd" 


def get_airly_data():
    api_url = "https://airapi.airly.eu/v2/measurements/nearest?lat=5.6502465&lng=-0.1832575&maxDistanceKM=5&maxResults=3"
    response = requests.get(api_url, headers={'apikey': apikey})
    data = json.load(StringIO(response.text))
    #keys for airly data are ['current', 'forecast', 'history']
    # pm25conc = data['current']['values'][1]['value']
    # pm10conc = data['current']['values'][2]['value']
    # humidity = data['current']['values'][4]['value']
    # tempurature = data['current']['values'][5]['value']
    # final_data = pm25conc, pm10conc, humidity, tempurature
    # return final_data
    return data['history']



if __name__ == "__main__":
    dat = get_airly_data()
    print(dat)


    
# print(df.head())