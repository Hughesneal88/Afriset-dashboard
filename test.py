import pickle
import pandas as pd

with open('data', 'rb') as f: 
    data = pickle.load(f)


print(data)

df = data
# print(data[0:10])

# for i in data[0:10]:
#     print(i["Datetime"],i["b'PM25'"], i["b'PM10'"])

# dat = {"date": i["Datetime"] for i in data[0::] "pm25":[i["b'PM25'"] for i in data[0::]] "pm10":[i["b'PM10'"] for i in data[0::]]}
dat = {
    "date": [i["Datetime"] for i in df],
    "pm25": [i["b'PM25'"] for i in df],
    "pm10": [i["b'PM10'"] for i in df],
}

df = pd.DataFrame(dat)
df.index = df["date"]

minimal =min(df.index)
print(minimal)