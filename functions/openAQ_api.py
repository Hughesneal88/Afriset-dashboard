from unicodedata import normalize
import requests
import pandas as pd
import json
from io import StringIO
import pickle

api_key = open("OpenAQ api", 'r').readline()  # Your API key here.

afriset_location_id = ["2453497","2453501", "2453500"]
# for i in afriset_location_id:
#     params = {"locationId": i}

#     res = requests.get("https://api.openaq.org/v2/measurements", headers={"X-API-Key": api_key}, params=params)
#     with open('openaq-afriset.json', 'a') as f:
        # f.write(res.text)
# for i in range(1,2):
#     params = {
#         "location": "Afri-SET",
#         "date_from": "2024-10-01T12:00:03.604475",
#         "date_to": "2024-10-24T12:00:03.604475",
#         "sort": "asc",
#         "page": i,
#         "offset": "0",
#         "limit": "10000"
#         }

#     res = requests.get("https://api.openaq.org/v2/measurements", headers={"X-API-Key": api_key}, params=params)
#     with open('openaq-afriset-1.json', 'wb') as f:
#         pickle.dump(res.json(), f)

with open('openaq-afriset-1.json', 'rb') as f:
    raw = pickle.load(f)
    # raw = raw[0]
    # print(raw)
    df = pd.DataFrame(pd.json_normalize(raw))



real_df = pd.DataFrame(df["results"][0])
# real_df.index = real_df['date']
for i in real_df['date'].index:
    real_df.loc[i, "date"] = real_df["date"][i]["utc"]
real_df.index = real_df["date"]
# print(real_df)

filtered_df = real_df[real_df['locationId'] == 2453500]
print(filtered_df)

filtered_df.index = pd.to_datetime(filtered_df.index)

with open('openaq-collins.pkl', 'wb') as f:
    pickle.dump(filtered_df, f)



# res = requests.get("https://api.openaq.org/v2/locations?limit=1000&page=1&offset=0&sort=desc&country=GH&order_by=lastUpdated&dump_raw=false", headers={"X-API-Key": api_key})
