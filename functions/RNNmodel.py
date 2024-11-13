import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import time


# functions/get_t640.py
import json
import os
from pycampbellcr1000 import CR1000
import datetime
import pickle
import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # Import for loading the RNN model
from threading import Thread
from datetime import datetime, timedelta
from functions.graph_plot import return_df

from functions.get_t640 import get_T640

# Global variables for the RNN model and scaler
rnn_model = None
scaler = MinMaxScaler()
predictions_buffer = []  # Temporary storage for predictions

def load_historical_data():
    try:
        with open('data/T640_and_MET_data_forecast.pkl', 'rb') as f:
            historical_data = pickle.load(f)
        
        # Convert to DataFrame if it's not already
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

# Function to create and train the RNN model
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Use Input layer to define the input shape
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load historical data
historical_data = load_historical_data()
if historical_data is not None and not historical_data.empty:
    X, y, scaler = prepare_data(historical_data)

    # Check if X and y are not empty
    if X.size > 0 and y.size > 0:
        # Create and train the model
        rnn_model = create_rnn_model((X.shape[1], X.shape[2]))  # Input shape is (time_steps, features)
        rnn_model.fit(X, y, epochs=10, batch_size=32, verbose=1)

        # Save the trained model
        rnn_model.save('updated_rnn_model.h5')
    else:
        print("Prepared data is empty. Cannot train the model.")
else:
    print("Historical data is empty or could not be loaded.")

# Function to prepare data for RNN
def prepare_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Function to update the RNN model with new data
def update_forecast(new_data):
    global rnn_model
    # Scale the new data
    if new_data.size == 0:
        print("No data available for scaling.")
        return  # Exit the function if there's no data

    new_data_scaled = scaler.fit_transform(new_data.reshape(-1, 1))
    
    # Prepare the data for RNN
    X, y = prepare_data(new_data_scaled, time_steps=10)  # Adjust time_steps as needed
    
    # Debugging: Print the shape of X and y
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Ensure this is valid

    if rnn_model is None:
        rnn_model = create_rnn_model((X.shape[1], 1))  # Create model if it doesn't exist
    rnn_model.fit(X, y, epochs=10, batch_size=32, verbose=1)  # Train the model

# New function to forecast using the RNN model
def forecast_with_rnn(df):
    global rnn_model, scaler  # Ensure we can access the global RNN model and scaler
    if rnn_model is None:
        raise Exception("RNN model is not trained yet.")

    # Prepare the data
    df = prepare_data(df.values, time_steps=10)  # Ensure df is a NumPy array
    df = pd.DataFrame(df)  # Convert back to DataFrame if needed

    # Get the current time and filter the last data point
    current_time = pd.to_datetime(datetime.datetime.now())  # Get the current time
    last_data = df.iloc[-1:]  # Get the last row of data

    # Generate forecast times starting from the next hour
    forecast_times = [current_time + pd.Timedelta(hours=i+1) for i in range(5)]  # Next 5 hours

    # Prepare input for RNN
    last_data_scaled = scaler.transform(last_data)  # Scale the last data point
    last_data_scaled = last_data_scaled.reshape((1, last_data_scaled.shape[0], last_data_scaled.shape[1]))  # Reshape for LSTM

    # Make predictions
    predictions = []
    for _ in range(5):  # Forecast for the next 5 hours
        pred = rnn_model.predict(last_data_scaled)
        predictions.append(pred[0][0])  # Store the prediction

        # Prepare the next input data
        new_row = {
            'Hour': (last_data['Hour'].values[0] + 1) % 24,  # Increment hour
            'Day': last_data['Day'].values[0],
            'PM25_lag1': pred[0][0],  # Use the predicted PM2.5 as lagged feature
            'RH_lag1': last_data['RH_lag1'].values[0],
            'Temp_lag1': last_data['Temp_lag1'].values[0],
            'WD_lag1': last_data['WD_lag1'].values[0],
            'WS_lag1': last_data['WS_lag1'].values[0],
            'Season': last_data['Season'].values[0],
        }
        last_data = pd.DataFrame([new_row])  # Update current data for the next prediction
        last_data_scaled = scaler.transform(last_data)  # Scale the new data
        last_data_scaled = last_data_scaled.reshape((1, last_data_scaled.shape[0], last_data_scaled.shape[1]))  # Reshape for LSTM

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({
        'Datetime': forecast_times,
        'PM25': predictions,
        'AQI': [calculate_aqi(pm25) for pm25 in predictions]  # Calculate AQI for each prediction
    })

    # Set the Datetime as the index (optional)
    forecast_df.set_index('Datetime', inplace=True)
    forecast_df.index = forecast_df.index.strftime('%I:%M %p')

    return forecast_df

# Function to continuously update the RNN model with new data
# def continuous_forecasting():
#     while True:
#         df = return_df()  # Get the latest data frame
#         # Use the new forecast_with_rnn function for predictions
#         forecast_values = forecast_with_rnn(df)  # Calculate the forecast using the RNN
#         save_forecast(forecast_values)  # Save the forecast data to a file or database
#         update_forecast(forecast_values)  # Update the RNN model with new data
#         time.sleep(3600)  # Update every hour

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
    
def prepare_data(df):
    # Ensure the index is a DatetimeIndex
    # df.index = pd.to_datetime(df.index)  # Convert index to datetime if not already

    # Drop rows with NaN values
    # df.dropna(inplace=True)
    df= prepare_df(df)

    # Select relevant features for training
    features = df[['PM25', 'Humidity', 'Temperature', 'WD', 'WS']]  # Adjust based on your data
    target = df['PM25']  # Assuming PM25 is the target variable

    # Fit the MinMaxScaler on the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare the data for RNN
    X, y = [], []
    time_steps = 10  # Number of time steps to look back
    for i in range(len(features_scaled) - time_steps):
        X.append(features_scaled[i:(i + time_steps)])
        y.append(target.iloc[i + time_steps])  # Target is the next value after the time steps

    return np.array(X), np.array(y), scaler
    
def prepare_df(df):
    # df['Target'] = df['PM25']
    df.dropna(inplace=True)
    print('Index for forecast: ', df.index)
    # df.set_index('Datetime', inplace=True)
    print('columns for forecast: ', df.columns)

    df.index = pd.to_datetime(df.index) 
    df['Hour'] = df.index.hour
    df['Hour'] = df['Hour']
    df['PM25_lag1'] = df['PM25']
    # df['PM2.5_AQI_lag1'] = df['NowCast_PM2.5_AQI'].shift(-1)
    df[['RH_lag1', 'Temp_lag1', 'WD_lag1', 'WS_lag1']] = df[['Humidity', 'Temperature', 'WD', 'WS']]
    df['Season'] = [get_season(date) for date in df.index]
    df['Day'] = df.index.dayofweek

    df.dropna(inplace=True)
    return df

# Load the model at the start
def load_rnn_model():
    global rnn_model, scaler
    if os.path.exists('updated_rnn_model.h5'):
        rnn_model = load_model('updated_rnn_model.h5')
    else:
        print("Model file not found. Creating and training a new model with historical data.")
        try:
            # Load historical data from pickle
            with open('data/T640_and_MET_data_forecast.pkl', 'rb') as f:
                historical_data = pickle.load(f)
            
            # Convert to DataFrame if it's not already
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)
            
            # Prepare the data
            prepared_data = prepare_df(historical_data)
            
            # Select features
            features = prepared_data[['PM25_lag1', 'RH_lag1', 'Temp_lag1', 'WD_lag1', 'WS_lag1', 
                                    'Hour', 'Day', 'Season']]
            target = prepared_data['PM25']  # Current PM25 values as target
            
            # Scale the features
            features_scaled = scaler.fit_transform(features)
            X, y = prepare_data(features_scaled, time_steps=10)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Create and train the model
            rnn_model = create_rnn_model()
            rnn_model.fit(X, y, epochs=10, batch_size=32, verbose=1)
            
            # Save the initial model
            rnn_model.save('updated_rnn_model.h5')
            print("Initial model trained and saved successfully.")
            
        except Exception as e:
            print(f"Error creating initial model: {e}")
            # Create an untrained model as fallback
            rnn_model = create_rnn_model()

# Function to check if the model is trained
def is_model_trained():
    return rnn_model is not None and hasattr(rnn_model, 'layers') and len(rnn_model.layers) > 0

# Update forecast function
def update_forecast():
    while True:
        df = return_df()  # Get the latest data frame
        if df.empty:
            print("No data available for forecasting.")
            time.sleep(3600)
            continue
        
        try:
            if not is_model_trained():
                print("RNN model is not trained yet. Training the model.")
                train_model()  # Train the model if it is not trained
            
            # Prepare the data for forecasting
            features_scaled = scaler.transform(df.values)  # Scale the features
            
            # Debugging: Print the shape of the scaled features
            print("Scaled features shape:", features_scaled.shape)
            
            X, y = prepare_data(features_scaled, time_steps=10)  # Ensure time_steps is passed correctly
            
            # Debugging: Print the shape of X and y
            print("X shape:", X.shape)
            print("y shape:", y.shape)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            forecast_values = forecast_with_rnn(df)  # Calculate the forecast using the RNN
            predicted_value = forecast_values['PM25'].iloc[-1]  # Get the last predicted value
            predictions_buffer.append(predicted_value)  # Store the prediction
            
            # Optionally, save the forecast values
            save_forecast(forecast_values)
        except Exception as e:
            print(f"Error during forecasting: {e}")
        
        time.sleep(3600)  # Update every hour

# Load the model at the start
load_rnn_model()


# Function to save actual vs predicted values
def save_actual_vs_predicted(actual, predicted):
    current_time = datetime.now()
    
    # Load the latest data from pickle
    try:
        with open('data/T640_and_MET_data_forecast.pkl', 'rb') as f:
            latest_data = pickle.load(f)
            
        df = pd.DataFrame({
            'Datetime': [current_time],
            'Actual': [actual],
            'Predicted': [predicted],
            'PM25_lag1': [predicted],  # Using predicted as lag for next prediction
            'RH_lag1': [latest_data['Humidity']],  # Get humidity from pickled data
            'Temp_lag1': [latest_data['Temperature']],  # Get temperature from pickled data
            'WD_lag1': [latest_data['WD']],  # Get wind direction from pickled data
            'WS_lag1': [latest_data['WS']],  # Get wind speed from pickled data
            'Hour': [current_time.hour],
            'Day': [current_time.weekday()],
            'Season': [get_season(current_time)]
        })
        
        df.to_csv('actual_vs_predicted.csv', mode='a', 
                  header=not os.path.exists('actual_vs_predicted.csv'), 
                  index=False)
    except Exception as e:
        print(f"Error accessing pickled data: {e}")

# Function to train the model
def train_model():
    global rnn_model, scaler
    historical_data = pd.read_csv('actual_vs_predicted.csv')
    
    # Prepare the data for training
    features = historical_data[['PM25_lag1', 'RH_lag1', 'Temp_lag1', 'WD_lag1', 'WS_lag1', 
                              'Hour', 'Day', 'Season']]
    target = historical_data['Actual']
    
    # Scale the features
    features_scaled = scaler.fit_transform(features)
    X, y = prepare_data(features_scaled, time_steps=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train the model
    rnn_model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    rnn_model.save('updated_rnn_model.h5')

# Background training function
def background_training():
    while True:
        # Check if an hour has passed since the last prediction
        if predictions_buffer:
            # Get the actual value from the DataFrame or another source
            actual_value = get_T640()  # Implement this function to retrieve the actual value
            
            if actual_value is not None:
                # Retrieve the last prediction
                last_prediction = predictions_buffer.pop(0)  # Get the first prediction
                save_actual_vs_predicted(actual_value, last_prediction)  # Save actual vs predicted
                
                # Train the model with the new data
                train_model()
        
        time.sleep(3600)  # Check every hour


# Start the training thread
training_thread = Thread(target=background_training)
training_thread.daemon = True
training_thread.start()

# # Entry point for the forecasting process
# if __name__ == "__main__":
#     continuous_forecasting()

def prepare_data(df):
    # Ensure the index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)  # Convert index to datetime if not already

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Select relevant features for training
    features = df[['PM25', 'Humidity', 'Temperature', 'WD', 'WS']]  # Adjust based on your data
    target = df['PM25']  # Assuming PM25 is the target variable

    # Fit the MinMaxScaler on the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare the data for RNN
    X, y = [], []
    time_steps = 10  # Number of time steps to look back
    for i in range(len(features_scaled) - time_steps):
        X.append(features_scaled[i:(i + time_steps)])
        y.append(target.iloc[i + time_steps])  # Target is the next value after the time steps

    return np.array(X), np.array(y), scaler
