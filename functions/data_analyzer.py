import json
from get_city import time_elapsed

#for praxius
#url = https://aws.southcoastscience.com/topicMessages?topic=south-coast-science-demo%2Fbrighton%2Floc%2F1%2Fclimate&startTime=2021-01-23T09%3A22%3A50.460Z&endTime=2021-01-23T09%3A23%3A50.460Z
#api_key = 'south-coast-science-test'

def get_praxius_data():
    pass



def get_pm25(obj):
    # Get pm2.5 values from json response from API
    val = json.loads(obj)
    # aquis = val["hourly"]["p2"]["aqius"]
    # conc = val["hourly"]["p2"]["conc"]
    given_time = obj['hourly']['ts']
    passed_time = round(time_elapsed(given_time).seconds/1000))
    # return aquis, conc
    return city, country, aqius, passed_time

def get_pm10(obj):
    # Get pm10 values from the json response from API
    val = json.loads(obj)
    aquis = val["hourly"]["p1"]["aquis"]
    conc = val["hourly"]["p2"]["conc"]
    return aquis, conc

def get_co2(obj):
    # Get pm10 values from the json response from API
    val = json.loads(obj)
    aquis = val["hourly"]["co"]["aquis"]
    conc = val["hourly"]["co"]["conc"]
    return aquis, conc



# "hourly": [

#    {

#     "p2_sum": 119, //total PM2.5 concentration within 1 hour

#     "p2_count": 4, //number of measurements taken within 1 hour

#     "p1_sum": 130, //total PM10 concentration within 1 hour

#     "p1_count": 4, //number of measurements taken within 1 hour

#     "co_sum": 1809, //total CO2 concentration within 1 hour

#     "co_count": 4, //number of measurements taken within 1 hour

#     "hm_sum": 95,

#     "hm_count": 4,

#     "tp_sum": 20.24477,

#     "tp_count": 4,

#     "ts": "2017-12-08T02:00:00.000Z",

#     "outdoor_station": { //refer to the documentation link above

#      "ts": "2017-12-06T15:00:00.000Z",

#      "aqius": 41,

#      "mainus": "p2",

#      "aqicn": 14,

#      "maincn": "p2",

#      "p2": {

#       "conc": 10,

#       "aqius": 41,

#       "aqicn": 14

#      },

#      "api_id": "4395"

#     },

#     "outdoor_weather": {

#      "ts": "2017-12-06T15:00:00.000Z",

#      "tp": 4,

#      "pr": 1025,

#      "hu": 29,

#      "ws": 0,

#      "wd": 327,

#      "ic": "01n"

#     }

#    },