import coreapi.client
import requests
import json
import coreapi

# Sensor1: 10676715
# Sensor2: 126620191
# Sensor3: 10667948

api_url = "http://api.sensors.africa/v2/data"
# def get_token():
#     url="http://api.sensors.africa/v1/get-auth-token/"
#     # payload = {
#     #     'Token':'test-afriset'
#     # }
#     payload = {
#         'username': 'hughesneal88',
#         'password': 'Password#123'
#     }
#     final = json.dumps(payload)
#     headers = {
#         'Content-Type': "application/json",
#     }
#     requests.get(url, data=final, headers=headers)
# def get_sensor_data():
#     response = requests.get(api_url)
#     data = json.loads(response.text)
#     return data

# print(get_sensor_data())
# print(get_token())
# client = coreapi.Client(auth=coreapi.auth.TokenAuthentication(token='test-afriset', scheme='Bearer' ))
# print(coreapi.auth.TokenAuthentication(token='Afriset').token)
# # print(coreapi.Client.action(api_url, document='http://api.sensors.africa/docs/',keys='data', action=['get']))
# print(coreapi.Link(api_url, action='get'))
# print(coreapi.Client.get(client, url=api_url))

# Initialize a client & load the schema document
client = coreapi.Client()
schema = client.get("http://api.sensors.africa/docs/")

# Interact with the API endpoint
action = ["get-auth-token", "create"]
params = {
    'token': 'test-afriset',
}
result = client.action(schema, action, params=params)

# Interact with the API endpoint
action = ["v2", "data &gt; list"]
params = {
    "next_page": ...,
    "sensor": ...,
    "sensor__public": ...,
    "location__country": ...,
    "location__city": ...,
    "timestamp__gte": ...,
    "timestamp__lte": ...,
}
result = client.action(schema, action, params=params)
