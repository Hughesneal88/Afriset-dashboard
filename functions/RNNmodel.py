import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta

# Load the dataset
def load_historical_data():
    try:
        with open('data/T640_and_MET_data_forecast.pkl', 'rb') as f:
            historical_data = pickle.load(f)
        
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

data = load_historical_data()

# Preprocess the data
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
# Rename columns
data.rename(columns={'Temp': 'Temperature', 'RH': 'Humidity'}, inplace=True)
print(data.columns)

values = data['PM25'].values.reshape(-1,1)  # Predicting PM25

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 5  # Using the previous 10 hours to predict the next hour
X, y = create_sequences(scaled_values, SEQ_LENGTH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained model
def load_model():
    try:
        model = tf.keras.models.load_model('models/updated_rnn_model.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    print("Pre-trained model not found. Please ensure the model is available at 'models/updated_rnn_model.h5'.")
    print("Training a new model...")
    
    # Define the RNN model
    def create_rnn_model(input_shape):
        model = Sequential()
        model.add(SimpleRNN(100, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(SimpleRNN(100, activation='relu', return_sequences=True))
        model.add(SimpleRNN(50, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    # Build and train the model
    model = create_rnn_model((SEQ_LENGTH, 1))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save('models/updated_rnn_model.h5')

# Evaluate the model on the training set
train_loss = model.evaluate(X_train, y_train)
print(f'Training Loss: {train_loss}')

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Evaluate the model on the training set
train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train)

train_r2 = r2_score(y_train_actual, train_predictions)
print(f'Training R^2 Score: {train_r2}')

# Evaluate the model on the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test)

test_r2 = r2_score(y_test_actual, test_predictions)
print(f'Test R^2 Score: {test_r2}')

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate error metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# # Plot the predicted vs actual values
# plt.figure(figsize=(10, 6))
# plt.plot(y_test, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('PM25')
# plt.title('Actual vs Predicted PM25')
# plt.legend()
# plt.savefig('actual_vs_predicted.png')

# Save predictions to a CSV file
# pred_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
# pred_df.to_csv('predictions.csv', index=False)

# Function to forecast using the RNN model
def forecast_with_rnn(df):
    global model, scaler
    if model is None:
        raise Exception("RNN model is not trained yet.")

    df = df.values.reshape(-1, 1)
    df_scaled = scaler.transform(df)
    X, _ = create_sequences(df_scaled, SEQ_LENGTH)
    
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    forecast_times = [datetime.now() + timedelta(hours=i+1) for i in range(5)]
    
    forecast_df = pd.DataFrame({
        'Datetime': forecast_times,
        'PM25': predictions.flatten()[:5],
        'AQI': [calculate_aqi(pm25) for pm25 in predictions.flatten()[:5]]
    })
    
    forecast_df.set_index('Datetime', inplace=True)
    forecast_df.index = forecast_df.index.strftime('%I:%M %p')
    
    return forecast_df

# Function to calculate AQI
def calculate_aqi(pm25):
    if pm25 < 0:
        return None
    elif pm25 <= 9.0:
        return (50 / 9.0) * pm25
    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 9.1)) * (pm25 - 9.1) + 51
    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 125.4:
        return ((200 - 151) / (125.4 - 55.5)) * (pm25 - 55.5) + 151
    elif pm25 <= 225.4:
        return ((300 - 201) / (225.4 - 125.5)) * (pm25 - 125.5) + 201
    elif pm25 <= 500.4:
        return ((500 - 301) / (500.4 - 225.5)) * (pm25 - 225.5) + 301
    else:
        return 500
