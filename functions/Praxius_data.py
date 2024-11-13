import json
import csv
import requests
from datetime import datetime, timedelta

### formula for USAQI
# Ip = [IHi – ILo / BPHi – BPLo] (Cp – BPLo) + ILo 


# Ip = index of pollutant p
# Cp = truncated concentration of pollutant p
# BPHi = concentration breakpoint i.e. greater than or equal to Cp
# BPLo = concentration breakpoint i.e. less than or equal to Cp
# IHi = AQI value corresponding to BPHi
# ILo = AQI value corresponding to BPLo

def get_praxius_data():
    """Get data from Praxius API"""
    
    # Set up the URL and headers for the GET request
    url = "https://aws.southcoastscience.com/topicMessages?topic=south-coast-science-test/afri-set/loc/868/particulates&startTime=%sZ&endTime=%sZ" %((datetime.now()-timedelta(minutes=1)).isoformat(), datetime.now().isoformat())
    headers = {'Authorization': 'api_key south-coast-science-test'}

    # Send the HTTP request
    response = requests.get(url, headers=headers)
    d = response.text
    # with open('data.csv', 'w') as file:
    #     d = response.text
    #     json_data = json.loads(d)['Items']
    #     # file.write(json.dumps(json_data))
    #     csv_writer = csv.writer(file)
 
    #     count = 0
    #     for data in json_data:
    #         if count == 0:
    #             header = data.keys()
    #             csv_writer.writerow(header)
    #             count += 1
    #         csv_writer.writerow(data.values())
    # If the request is successful, return the JSON object
    json_data = json.loads(d)['Items']
    pm2p5 = json_data[0]['val']['pm2p5']
    pm10 = json_data[0]['val']['pm10']
    pm1 = json_data[0]['val']['pm1']
    humidity = json_data[0]['val']['sht']['hmd']
    temp = json_data[0]['val']['sht']['tmp']
    return {"PM2.5":  pm2p5, "PM10" : pm10, "PM1" : pm1, "Humidity": humidity, "Tempurature" : temp}

# def calculate_us_aqi(pm25):
#     """Calculate US Air Quality Index based on PM2.5 concentration."""

#     assert pm25 >= 0, "PM2.5 must be non-negative!"

#     if pm25 < 12:
#         ip = 0
#     elif pm25 < 18.5:
#         ip = 1
#     else:
#         ip = 2

#     cp = min(max(float(pm25), 0), 400)   # Truncate values outside valid range

#     bphi = [0, 12, 18.5]                                 # Concentration Breakpoints
#     ihi = [0, 50, 100]                                                                     # AQI Values at BP Hi
#     ihi = [0, 50, 100]                                                                     # AQI Values at BPs


if __name__ == "__main__":
    url = "https://aws.southcoastscience.com/topicMessages?topic=south-coast-science-test/afri-set/loc/868/particulates&startTime=%sZ&endTime=%sZ" %((datetime.now()-timedelta(minutes=1)).isoformat(), datetime.now().isoformat())
    headers = {'Authorization': 'api_key south-coast-science-test'}
    response = requests.get(url, headers=headers)
    d = response.text
    # with open('data.csv', 'w') as file:
    #     d = response.text
    #     json_data = json.loads(d)['Items']
    #     csv_writer = csv.writer(file)
 
    #     count = 0
    #     for data in json_data:
    #         if count == 0:
    #             header = data.keys()
    #             csv_writer.writerow(header)
    #             count += 1
    #         csv_writer.writerow(data.values())
    #     # file.write(json.dumps(json_data))
    # # print(get_praxius_data())
    #         print(data.values())
    print(json.dumps(json.loads(response.text)['Items'][0]['val']))