import json
import os
from pycampbellcr1000 import CR1000
import datetime
import pickle
import pandas as pd
import joblib

# start = datetime.datetime.now()-datetime.timedelta(minutes=1)
# end = datetime.datetime.now()



def get_T640(city, country):
    sensor_url = get_sensor_url(city, country)
    if os.path.isfile("T640"):
        with open("T640", 'rb') as f:
            dat = pickle.load(f)
            if datetime.datetime.now() - dat["Time"] < datetime.timedelta(hours=1):
                data = [val for val in dat.values()]
                return data
            else:
                device = CR1000.from_url(sensor_url)
                start = datetime.datetime.now().replace(second=0, microsecond=0) - datetime.timedelta(hours=1)
                end = datetime.datetime.now()
                weather = device.get_data('OneHour_MET', start, end)
                particulates = device.get_data('OneHour_T640', start, end)
                if particulates == []:
                    particulates = device.get_data('OneMin_T640', start, end)
                    pm25_df = pd.DataFrame(particulates)
                    pm25_df_reindex = pm25_df.set_index('Datetime')
                    pm25_df_reindex = pm25_df_reindex.resample('h').mean()
                    time = pm25_df_reindex.index[1]
                    pm25 = round(pm25_df_reindex["b'PM2.5'"].iloc[1], 2)
                    pm10 = round(pm25_df_reindex["b'PM10'"].iloc[1], 2)
                    aqi = int(pm25_df_reindex["b'NowCast_PM2.5_AQI'"].iloc[1])
                else:
                    time = particulates[0]['Datetime']
                    pm25 = round(particulates[0]["b'PM25'"], 2)
                    pm10 = round(particulates[0]["b'PM10'"], 2)
                    aqi = int(particulates[0]["b'NowCast_PM2.5_AQI'"])
                temp = weather[0]["b'Temp'"]
                humidity = weather[0]["b'RH'"]
                w_icon = 0
                data = {"city": city, "country": country, "AQI": aqi, "Time": time, "Temp": temp, "w_icon": 0, "Humidity": humidity, "PM2.5": pm25}
                with open("T640", "wb") as f:
                    pickle.dump(data, f)
                data = [val for val in data.values()]
                device.bye()
    return data

def ip_info(addr=None):
    from urllib.request import urlopen
    import json
    if addr is None:
        url = 'https://ipinfo.io/json'
    else:
        print(addr)
        url = 'https://ipinfo.io/' + addr + '/json'
    # if res==None, check your internet connection
    res = urlopen(url)
    data = json.load(res)
    print(data)
    if 'city' in data:
        city = data['city']
        country = data['country']
        return city, country
    else:
        return None


def get_sensor_url(city, country):
    # Define a mapping of city and country to sensor URLs
    sensor_mapping = {
        ("Accra", "Ghana"): 'tcp:41.190.69.252:6785',
        # Add more mappings as needed
    }
    
    # If city is None, default to Accra, Ghana
    if city or country is None:
        city = "Accra"
        country = "Ghana"
    
    return sensor_mapping.get((city, country), 'default_sensor_url')
# return city, country, aqi, time, temp, w_icon, humidity, pm25, data


def prepare_data(df):
    # df['Target'] = df['PM25']
    df.dropna(inplace=True)
    print('Index for forecast: ', df.index)
    # df.set_index('Datetime', inplace=True)
    print('columns for forecast: ', df.columns)

    df['Hour'] = df.index.hour
    df['Hour'] = df['Hour']
    df['PM25_lag1'] = df['PM25']
    # df['PM2.5_AQI_lag1'] = df['NowCast_PM2.5_AQI'].shift(-1)
    df[['RH_lag1', 'Temp_lag1', 'WD_lag1', 'WS_lag1']] = df[['Humidity', 'Temperature', 'WD', 'WS']]
    df['Season'] = [get_season(date) for date in df.index]
    df['Day'] = df.index.dayofweek

    df.dropna(inplace=True)
    return df

def forecast_next_hours(model, last_data, hours=5):
    predictions = []
    current_data = last_data.copy()

    for _ in range(hours):
        # Make prediction
        pred = model.predict(current_data)
        predictions.append(pred[0])  # Store the prediction
        
        # Prepare the next input data
        new_row = {
            'Hour': (current_data['Hour'].values[0] + 1) % 24,  # Increment hour
            'Day': current_data['Day'].values[0],
            'PM25_lag1': pred[0],  # Use the predicted PM2.5 as lagged feature
            'RH_lag1': current_data['RH_lag1'].values[0],
            'Temp_lag1': current_data['Temp_lag1'].values[0],
            'WD_lag1': current_data['WD_lag1'].values[0],
            'WS_lag1': current_data['WS_lag1'].values[0],
            'Season': current_data['Season'].values[0],
        }
        current_data = pd.DataFrame([new_row])  # Update current data for the next prediction

    return predictions

def calculate_aqi(pm25):
    if pm25 < 0:
        return None  # Invalid PM2.5 value
    elif pm25 <= 9.0:
        return (50 / 9.0) * pm25  # Good
    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 9.1)) * (pm25 - 9.1) + 51  # Moderate
    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101  # Unhealthy for Sensitive Groups
    elif pm25 <= 125.4:
        return ((200 - 151) / (125.4 - 55.5)) * (pm25 - 55.5) + 151  # Unhealthy
    elif pm25 <= 225.4:
        return ((300 - 201) / (225.4 - 125.5)) * (pm25 - 125.5) + 201  # Very Unhealthy
    elif pm25 <= 500.4:
        return ((500 - 301) / (500.4 - 225.5)) * (pm25 - 225.5) + 301  # Hazardous
    else:
        return 500  # AQI capped at 500
    
def get_season(date):
    if date.month in [11, 12, 1, 2]:
        return 1
    elif date.month in [3,4]:
        return 2
    else:
        return 3
    
def get_forecast_df():
    data_file = "data/T640_and_MET_data.pkl"
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
            return df
    else:
        print("T640_and_MET_data.pkl file not found in the data folder.")
        return pd.DataFrame()  # Return an empty DataFrame if the file does not exist
            

def forecast(df):
    model = joblib.load('models/regression_model.joblib')
    FEATURES = ['Hour', 'Day', 'PM25_lag1', 'RH_lag1', 'Temp_lag1', 'WD_lag1', 'WS_lag1','Season'] 
    TARGET = 'PM25'

    df = prepare_data(df)
    df = df[FEATURES]
    
    # Get the current time and filter the last 5 hours of data
    # if df.empty:
    #     print("DataFrame is empty. Cannot calculate forecast.")
    #     return None  # Handle this case as needed
    current_time = df.index[-1]  # Get the last timestamp from the dataframe
    last_data = df[(df.index >= current_time)]  # Filter for the last 5 hours
    print(last_data)

    # Generate forecast times starting from the next hour
    forecast_times = [current_time + pd.Timedelta(hours=i+1) for i in range(1, 6)]  # Next 5 hours excluding current hour
    predictions = forecast_next_hours(model, last_data)
    forecast_df = pd.DataFrame({
        'Datetime': forecast_times,
        'PM25': predictions,
        'AQI': [calculate_aqi(pm25) for pm25 in predictions]  # Calculate AQI for each prediction
    })

    # Set the Datetime as the index (optional)
    forecast_df.set_index('Datetime', inplace=True)
    forecast_df.index = forecast_df.index.strftime('%I:%M %p')


    return forecast_df

if __name__ == "__main__":
    print(get_T640())
