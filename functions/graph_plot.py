import os
import pickle
import pandas as pd
import plotly.graph_objs as go
import requests
from io import StringIO
from pycampbellcr1000 import CR1000
import datetime
import socket
from plotly.subplots import make_subplots
from plotly_calplot import calplot
import numpy as np
import logging
from .cache_utils import update_plot_cache, get_cached_plot

""" TODO: 
        1. add graphs for pm10, humidity, tempurature, add overlays to compare these things
        2. Add a tool to get graphs based on a range and checkboxes to add parameters for the graphs
        3. Add a tool to get data based on a range and checkboxes to add parameters for the data
"""

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data_from_pickle(data_file):
    try:
        with open(data_file, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Pickle Error: {e}")
        return None
    except IOError as e:
        print(f"IO Error: {e}")
        return None

# import os
# import pickle
# import pandas as pd
# import datetime
# import socket
# from pycampbellcr1000 import CR1000

def get_plot_data(storage_format='pickle', hours=2):
    data_folder = "data"
    if storage_format == 'pickle':
        data_file = os.path.join(data_folder, "T640_and_MET_data.pkl")
    else:  # CSV
        data_file = os.path.join(data_folder, "T640_and_MET_data.csv")
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Define the columns we expect in our DataFrame
    expected_columns = ['Datetime', 'PM10', 'PM25', 'PM25_AQI', 'PM10_AQI', 'Temp', 'RH', 'WS', 'WD']

    # Load existing data or create new DataFrame
    if os.path.isfile(data_file):
        if storage_format == 'pickle':
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, pd.DataFrame):
                    df = data
                else:
                    print(f"Unexpected data type in pickle file: {type(data)}")
                    df = pd.DataFrame(columns=expected_columns)
        else:  # CSV
            df = pd.read_csv(data_file, parse_dates=['Datetime'])
    else:
        df = pd.DataFrame(columns=expected_columns)
        print("Created new DataFrame with expected columns")

    # # Set 'Datetime' as index for existing data
    # if 'Datetime' in df.columns:
    #     df['Datetime'] = pd.to_datetime(df['Datetime'])
    #     df.set_index('Datetime', inplace=True)
    # else:
    #     print("Error: 'Datetime' column not found in existing data")
    #     return None

    # Ensure 'Datetime' column is datetime type and set as index
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)  # Set Datetime as index
        last_datetime = df.index.max() if not df.empty else pd.Timestamp(datetime.datetime.now() - datetime.timedelta(days=7))
    else:
        print("No 'Datetime' column found in existing data")
        last_datetime = pd.Timestamp(datetime.datetime.now() - datetime.timedelta(days=7))

    print(f"Loaded data shape: {df.shape}")
    print(f"Columns in loaded data: {df.columns}")

    new_data_fetched = False
    # Check if we need to fetch new data
    if df.empty or datetime.datetime.now().replace(minute=0, second=0, microsecond=0) - last_datetime > datetime.timedelta(hours=1):
        try:
            # Attempt to connect to the device and fetch data
            print("Attempting to connect to device...")
            # socket.setdefaulttimeout(30)  # 30 seconds timeout
            device = CR1000.from_url('tcp:41.190.69.252:6785')  
            print("Device Connected successfully")
            
            start = last_datetime - datetime.timedelta(days=3)
            end = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
            print(f"Getting data from OneHour_T640 from {start} to {end}...")
            
            # Try to get air quality data from OneHour_T640
            aq_data = device.get_data('OneHour_T640', start, end)
            
            if not aq_data:
                print("No data from OneHour_T640, trying OneMin_T640...")
                start = last_datetime - datetime.timedelta(hours=hours)
                aq_data = device.get_data('OneMin_T640', start, end)
                if aq_data:
                    # Resample OneMin_T640 data to hourly
                    aq_df = pd.DataFrame(aq_data)
                    aq_df['Datetime'] = pd.to_datetime(aq_df['Datetime'])
                    aq_df.set_index('Datetime', inplace=True)
                    aq_df = aq_df.resample('H').mean()
                    aq_df.reset_index(inplace=True)
                else:
                    print("No data from OneMin_T640 either.")
                    aq_df = pd.DataFrame(columns=expected_columns)
            else:
                # Ensure aq_data is a DataFrame
                if isinstance(aq_data, list):
                    aq_df = pd.DataFrame(aq_data)
                elif isinstance(aq_data, pd.DataFrame):
                    aq_df = aq_data
                else:
                    print(f"Unexpected data type for aq_data: {type(aq_data)}")
                    aq_df = pd.DataFrame()  # Create an empty DataFrame if the type is unexpected

            # Strip 'b' prefix from column names
            aq_df.columns = [col.strip("b'") for col in aq_df.columns]

            # Rename columns for consistency
            column_mappings = {
                'PM25': ["PM25", "b'PM2.5'", "b'PM25'", "PM2.5"],
                'PM10': ["PM10", "b'PM10'"],
                # 'Temperature': ["Temperature", "b'Temp'", "Temp", "b'Temperature'"],
                # 'Humidity': ["Humidity", "b'RH'", "RH"],
                'PM25_AQI': ["NowCast_PM2.5_AQI", "b'NowCast_PM2.5_AQI'"],
                'PM10_AQI': ["NowCast_PM10_AQI", "b'NowCast_PM10_AQI'"],
                'WS': ["WS", "b'WS"],
                'WD': ["WD", "b'WD"]
            }

            for new_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in aq_df.columns:
                        aq_df.rename(columns={col: new_col}, inplace=True)
                        break

            for new_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        df.rename(columns={col: new_col}, inplace=True)
                        break

            # Debugging: Print columns from the device data
            print("Columns from device data after renaming:", aq_df.columns.tolist())

            met_data = device.get_data('OneHour_MET', start, end)
            # Ensure met_data is a DataFrame
            if isinstance(met_data, list):
                met_df = pd.DataFrame(met_data)
            elif isinstance(met_data, pd.DataFrame):
                met_df = met_data
            else:
                print(f"Unexpected data type for met_data: {type(met_data)}")
                met_df = pd.DataFrame()  # Create an empty DataFrame if the type is unexpected
            
            # Strip 'b' prefix from column names
            met_df.columns = [col.strip("b'") for col in met_df.columns]

            # Drop common columns except 'Datetime'
            common_columns = met_df.columns.intersection(aq_df.columns)
            columns_to_drop = [col for col in common_columns if col != 'Datetime']
            met_df.drop(columns=columns_to_drop, inplace=True)

            #Rename columns for consistency
            for new_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in met_df.columns:
                        met_df.rename(columns={col: new_col}, inplace=True)
                        break

            # Debugging: Print columns from the MET device data
            print("Columns from MET device data after renaming:", met_df.columns.tolist())

            # Merge new data with existing data on 'Datetime'
            if not aq_df.empty and not met_df.empty:
                new_df = pd.merge(aq_df, met_df, on='Datetime', how='outer')
                # new_df.reset_index(inplace=True)
                print('New DF columns: ', new_df.columns)
            elif not aq_df.empty:
                new_df = aq_df
            elif not met_df.empty:
                new_df = met_df
            else:
                new_df = pd.DataFrame(columns=expected_columns)

            # Align columns: Only append data for columns that exist in the existing data
            df.reset_index(inplace=True)
            # df.set_index('Datetime')
            # new_df.set_index('Datetime')
            common_columns = df.columns.intersection(new_df.columns)
            # df = df.combine_first(new_df[common_columns])
            print('Common Columns: ', common_columns)
            df = pd.concat([df, new_df], axis=0)
            df.set_index('Datetime', inplace=True)
            df.reset_index(inplace=True)

            # Remove duplicates and sort
            df.drop_duplicates(keep='last', inplace=True)
            # df = df[[common_columns]]
            print('Renamed df: ', df)
            # df.sort_index(inplace=True)

            # Reset index to make 'Datetime' a column again if needed
            # df.reset_index(inplace=True)

            # Save data only if new data is fetched
            if not new_df.empty:
                if storage_format == 'pickle':
                    with open(data_file, 'wb') as f:
                        pickle.dump(df, f)
                else:  # CSV
                    df.to_csv(data_file, index=False)
                print(f"Data Stored. Total records: {len(df)}")
                print(f"Final columns: {df.columns}")
                new_data_fetched = True
            else:
                print("No new data to store.")
        except socket.timeout:
            print("Connection timed out. The device is not responding. Using existing data.")
        except Exception as e:
            print(f"Error occurred while fetching data: {str(e)}. Using existing data.")
        finally:
            try:
                if 'device' in locals():
                    device.bye()  # Ensure the connection is closed properly
            except Exception as e:
                print(f"Error closing connection: {e}")
    else:
        print("Using existing data...")

    return data_file, new_data_fetched

def resample_data(data):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Ensure the index is a DatetimeIndex before resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: The index is not a DatetimeIndex. Attempting to convert.")
        df.index = pd.to_datetime(df.index)

    # List of columns we want to keep, with potential variations
    columns_to_keep = [
        'PM10', 'b\'PM10\'',
        'PM25', 'PM2.5', 'b\'PM25\'', 'b\'PM2.5\'',
        'PM10_AQI', 'b\'PM10_AQI\'',
        'PM25_AQI', 'b\'PM25_AQI\'',
        'NowCast_PM2.5_AQI', 'b\'NowCast_PM2.5_AQI\'',
        'NowCast_PM10_AQI', 'b\'NowCast_PM10_AQI\'',
        'RH', 'b\'RH\'','Humidity',
        'Temp', 'b\'Temp\'','Temperature',
        'WS', 'b\'WS\'', ''
        'WD', 'b\'WD\''
    ]
    
    # Keep only the columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]

    # Rename columns to remove 'b\'' prefix if present
    df.columns = [col.strip("b'") for col in df.columns]

    # Ensure PM2.5 column is named consistently
    if 'PM2.5' in df.columns:
        df.rename(columns={'PM2.5': 'PM25'}, inplace=True)

    # Rename 'Temp' to 'Temperature' if it exists
    if 'Temp' in df.columns:
        df.rename(columns={'Temp': 'Temperature'}, inplace=True)

    if 'RH' in df.columns:
        df.rename(columns={'RH': 'Humidity'}, inplace=True)

    if 'NowCast_PM2.5_AQI' or 'NowCast_PM25_AQI' in df.columns:
        if 'NowCast_PM2.5_AQI' in df.columns:
            df.rename(columns={'NowCast_PM2.5_AQI': 'PM25_AQI'}, inplace=True)
        elif 'NowCast_PM25_AQI' in df.columns:
            df.rename(columns={'NowCast_PM25_AQI': 'PM25_AQI'}, inplace=True)

    if 'NowCast_PM10_AQI' in df.columns:
        df.rename(columns={'NowCast_PM10_AQI': 'PM10_AQI'}, inplace=True)

    # Remove rows where all specified columns are NaN
    df = df.dropna(how='all')

    # Resample to hourly data and calculate the mean
    df_resampled = df.resample('h').mean()  # Ensure we take the mean for each hour

    # Reset index to make Datetime a column again
    df_resampled.reset_index(inplace=True)

    return df_resampled

def process_data(data_file_info, start_date=None, end_date=None, resample='h'):
    if isinstance(data_file_info, tuple):
        data_file, new_data_fetched = data_file_info
    else:
        data_file = data_file_info
        new_data_fetched = False

    data = load_data_from_pickle(data_file)
    if data is None or len(data) == 0:
        print("No data available in the pickle file.")
        return pd.DataFrame()  # Return an empty DataFrame if no data

    df = pd.DataFrame(data)
    if 'Datetime' not in df.columns:
        print("No 'Datetime' column found in the data.")
        return pd.DataFrame()  # Return an empty DataFrame if no Datetime column

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(f"Columns in loaded data: {df.columns}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Data shape before processing: {df.shape}")
    
    # Handle potential column name variations
    column_mappings = {
        'PM25': ["PM25", "b'PM2.5'", "b'PM25'", "PM2.5"],
        'PM10': ["PM10", "b'PM10'"],
        'Temperature': ["Temperature", "b'Temp'", "Temp", "b'Temperature'"],
        'Humidity': ["Humidity", "b'RH'", "RH"],
        'PM25_AQI': ["NowCast_PM2.5_AQI", "b'NowCast_PM2.5_AQI'"],
        'PM10_AQI': ["NowCast_PM10_AQI", "b'NowCast_PM10_AQI'"],
        'WS': ["WS", "b'WS"],
        'WD': ["WD", "b'WD"]
    }
    
    for new_col, possible_cols in column_mappings.items():
        for col in possible_cols:
            if col in df.columns:
                df.rename(columns={col: new_col}, inplace=True)
                break
        else:
            print(f"Warning: No {new_col} column found.")
    
    # Apply date filters if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        print(df.index.inferred_type)
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # upper_date_limit = df.index.min()
    # lower_date_limit = df.index.max()
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {numeric_columns}")
    
    # Resample numeric columns only
    if numeric_columns:

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        df_resampled = df[numeric_columns].resample(resample).mean()
        # print(f"Data shape after resampling: {df_resampled.shape}")
        # print(f"Resampled data head:\n{df_resampled.head()}")
        # print(f"Resampled date tail:\n{df_resampled.tail()}")
        return df_resampled.round(2)
    else:
        print("No numeric columns found for resampling.")
        return pd.DataFrame()  # Return an empty DataFrame if no numeric columns

def save_data_to_csv(data_file, start_date=None, end_date=None, resample='h', selected_columns=['PM25', 'PM25_AQI', 'PM10', 'PM10_AQI', 'Temperature', 'Humidity']):
    df = process_data(data_file, start_date, end_date, resample)
    
    # Define all available columns
    all_columns = ['PM25', 'PM25_AQI', 'PM10', 'PM10_AQI', 'Temperature', 'Humidity']
    
    # If no columns are selected, use all columns
    if not selected_columns:
        selected_columns = all_columns
    
    # Filter the dataframe to include only selected columns
    df_filtered = df[selected_columns]
    
    # Reset index to include datetime as a column
    df_filtered = df_filtered.reset_index()
    
    # Convert datetime to string in the desired format
    df_filtered['Datetime'] = df_filtered['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    csv_content = df_filtered.to_csv(index=False)
    return csv_content.encode('utf-8')



def hourly_T640(start_date=None, end_date=None, use_cache=True):
    # Set default start_date to one week ago if not provided
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    
    if end_date is None:
        end_date = datetime.datetime.now()
    
    cache_key = f"hourly_T640_{start_date}_{end_date}"
    
    if use_cache:
        cached_fig = get_cached_plot(cache_key)
        if cached_fig:
            print("Using cached plot")
            return cached_fig, None, None, False  # Assuming you want to return startDate and endDate as None for cached plots

    data_file, new_data_fetched = get_plot_data()
    # print(f"Data file: {data_file}, New data fetched: {new_data_fetched}")
    print(f"Data file: {data_file}, New data fetched: {new_data_fetched}")
    
    hourly = process_data(data_file, start_date=start_date, end_date=end_date, resample='H')
    upper_date_limit = process_data(data_file, start_date=None, end_date=None, resample='h').index.min()
    lower_date_limit = process_data(data_file, start_date=None, end_date=None, resample='h').index.max()
    print(f"Hourly DataFrame shape: {hourly.shape}")
    print(hourly.head())  # Check the first few rows of the DataFrame

    if hourly.empty:
        print("No data available for plotting.")
        return None, None, None, new_data_fetched
    
    # print(f"Hourly DataFrame shape: {hourly.shape}")  # Debugging line
    
    # hourly = filter_valid_pm_data(hourly)
    # if hourly.empty:
    #     print("No valid PM data available for plotting.")
    #     return None, None, None

    startDate = hourly.index.min()
    endDate = hourly.index.max()
    
    fig = go.Figure()

    if 'PM25' in hourly.columns:
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly['PM25'], mode='lines+markers', line=dict(color='red',width=2), name="PM2.5 (µg/m³)", yaxis='y'))
    
    if 'PM10' in hourly.columns:
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly['PM10'], mode='lines+markers', line=dict(color='blue',width=2), name="PM10 (µg/m³)", yaxis='y', visible='legendonly'))
        
    if 'PM25_AQI' in hourly.columns:
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly['PM25_AQI'], mode='lines+markers', line=dict(color='teal',width=2), name="PM2.5 AQI", yaxis='y'))
    
    if 'PM10_AQI' in hourly.columns:
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly['PM10_AQI'], mode='lines+markers', line=dict(color='green',width=2), name="PM10 AQI", yaxis='y', visible='legendonly'))
    
    # Calculate max value for PM
    max_pm = max(hourly['PM25'].max() if 'PM25' in hourly.columns else 0,
                 hourly['PM10'].max() if 'PM10' in hourly.columns else 0)

    # Set up the layout
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Hourly PM2.5 and AQI", font=dict(size=20)),
        xaxis_title=dict(text="Time", font=dict(size=20)),
                yaxis=dict(
            title=dict(text="Concentration (µg/m³)", font=dict(size=20)),
            side="left",
            range=[0, max_pm * 1.1] if max_pm > 0 else [0, 100]
        ),
        # yaxis=dict(
        #     title=dict(text="Concentration (µg/m³)", font=dict(size=20)),
        #     side="left",
        #     range=[0, max(hourly['PM25'].max(), hourly['PM10'].max()) * 1.1]
        # ),
        # yaxis2=dict(
        #     title=dict(text="AQI", font=dict(size=20)),
        #     side="right",
        #     overlaying="y",
        #     range=[0, max(hourly['PM25_AQI'].max(), hourly['PM10_AQI'].max()) * 1.1]
        # ),
        legend=dict(
            font=dict(size=12),
            bordercolor='black',
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(autorange=True) 
    if new_data_fetched:
        update_plot_cache(cache_key, fig)

    return fig, startDate.strftime('%Y-%m-%d %H:%M'), endDate.strftime('%Y-%m-%d %H:%M'), new_data_fetched

def return_df():
    data_file, new_data_fetched = get_plot_data()
    hourly = process_data(data_file, start_date=None, end_date=None, resample='H')
    df = hourly
    return df

def filter_valid_pm_data(df):
    # Filter out rows where both PM25 and PM10 are NaN or zero
    return df[(df['PM25'].notna() & (df['PM25'] != 0)) | (df['PM10'].notna() & (df['PM10'] != 0)) | (df['PM25_AQI'].notna() & (df['PM25_AQI'] != 0)) | (df['PM10_AQI'].notna() & (df['PM10_AQI'] != 0))]

def daily_T640(start_date=None, end_date=None, use_cache=True):
    cache_key = f"daily_T640_{start_date}_{end_date}"
    
    if use_cache:
        cached_fig = get_cached_plot(cache_key)
        if cached_fig:
            print("Using cached plot")
            return cached_fig, None, None, False  # Assuming you want to return startDate and endDate as None for cached plots

    data_file, new_data_fetched = get_plot_data()
    daily = process_data(data_file, start_date, end_date, resample='D')
    upper_date_limit = process_data(data_file, start_date=None, end_date=None, resample='D').index.min()
    lower_date_limit = process_data(data_file, start_date=None, end_date=None, resample='D').index.max()
    
    if daily.empty:
        print("No data available for plotting.")
        return None, None, None, new_data_fetched

    # Filter out rows where both PM25 and PM10 are NaN or zero
    # daily = filter_valid_pm_data(daily)

    # if daily.empty:
    #     print("No valid PM data available for plotting.")
    #     return None, None, None, new_data_fetched

    startDate = daily.index.min()
    endDate = daily.index.max()
    
    fig = go.Figure()

    # Add traces only for available data
    if 'PM25' in daily.columns:
        fig.add_trace(go.Scatter(x=daily.index, y=daily['PM25'], mode='lines+markers', line=dict(color='red',width=2), name="PM2.5 (µg/m³)", yaxis='y'))
    
    if 'PM10' in daily.columns:
        fig.add_trace(go.Scatter(x=daily.index, y=daily['PM10'], mode='lines+markers', line=dict(color='blue',width=2), name="PM10 (µg/m³)", yaxis='y', visible='legendonly'))
        
    if 'PM25_AQI' in daily.columns:
        fig.add_trace(go.Scatter(x=daily.index, y=daily['PM25_AQI'], mode='lines+markers', line=dict(color='teal',width=2), name="PM2.5 AQI", yaxis='y'))
    
    if 'PM10_AQI' in daily.columns:
        fig.add_trace(go.Scatter(x=daily.index, y=daily['PM10_AQI'], mode='lines+markers', line=dict(color='green',width=2), name="PM10 AQI", yaxis='y', visible='legendonly'))
    
    # Calculate max value for PM
    max_pm = max(daily['PM25'].max() if 'PM25' in daily.columns else 0,
                 daily['PM10'].max() if 'PM10' in daily.columns else 0)

    # Set up the layout
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Daily PM2.5 and PM10", font=dict(size=20)),
        xaxis_title=dict(text="Date", font=dict(size=20)),
        yaxis=dict(
            title=dict(text="Concentration (µg/m³)", font=dict(size=20)),
            side="left",
            range=[0, max_pm * 1.1] if max_pm > 0 else [0, 100]
        ),
        legend=dict(
            font=dict(size=12),
            bordercolor='black',
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(autorange=True) 

    if new_data_fetched:
        update_plot_cache(cache_key, fig)

    return fig, startDate, endDate, new_data_fetched

def create_air_quality_weather_plot(start_date=None, end_date=None, use_cache=True):
    cache_key = f"Air_Quality_Weather_{start_date}_{end_date}"
    
    if use_cache:
        cached_fig = get_cached_plot(cache_key)
        if cached_fig:
            print("Using cached plot")
            return cached_fig, None, None, False  # Assuming you want to return startDate and endDate as None for cached plots

    data_file, new_data_fetched = get_plot_data()
    upper_date_limit = process_data(data_file, start_date=None, end_date=None, resample='h').index.min()
    lower_date_limit = process_data(data_file, start_date=None, end_date=None, resample='h').index.max()
    hourly_df = process_data(data_file, start_date, end_date, resample='h')
    daily_df = process_data(data_file, start_date, end_date, resample='D')
    
    if hourly_df.empty and daily_df.empty:
        print("No data available for plotting.")
        return None, None, None, new_data_fetched
    
    # daily_df = filter_valid_pm_data(daily_df)
    # hourly_df = filter_valid_pm_data(hourly_df)

    # if hourly_df.empty:
    #     print("No Valid hourly PM data available for plotting.")
    #     return None, None, None, new_data_fetched
    
    # if daily_df.empty:
    #     print("No Valid daily PM data available for plotting.")
    #     return None, None, None, new_data_fetched
    
    # Create subplots: 2 rows, 1 column
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=False,
                        subplot_titles=("Hourly Data", "Daily Data"),
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                        vertical_spacing=0.1)

    # Colors for traces
    colors = {'PM25': 'red', 'PM10': 'blue', 'PM25_AQI': 'teal', 'PM10_AQI': 'green', 
              'Temperature': 'orange', 'Humidity': 'gray'}

    # Add traces for hourly and daily data
    for i, (df, row) in enumerate([(hourly_df, 1), (daily_df, 2)]):
        showlegend = i == 0  # Only show legend for the first (hourly) dataset
        fig.add_trace(go.Scatter(x=df.index, y=df['PM25'], name="PM2.5 (µg/m³)", mode="lines+markers",line=dict(color=colors['PM25']), legendgroup="PM25", showlegend=showlegend), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['PM10'], name="PM10 (µg/m³)", mode="lines+markers",line=dict(color=colors['PM10']), legendgroup="PM10", showlegend=showlegend, visible='legendonly'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['PM25_AQI'], name="PM2.5 AQI", mode="lines+markers", line=dict(color=colors['PM25_AQI']), legendgroup="PM25_AQI", showlegend=showlegend, visible='legendonly'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['PM10_AQI'], name="PM10 AQI", mode="lines+markers", line=dict(color=colors['PM10_AQI']), legendgroup="PM10_AQI", showlegend=showlegend, visible='legendonly'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], name="Temp (°C)", mode="lines+markers", line=dict(color=colors['Temperature']), legendgroup="Temperature", showlegend=showlegend), row=row, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Humidity'], name="Humidity (%)", mode="lines+markers", line=dict(color=colors['Humidity']), legendgroup="Humidity", showlegend=showlegend, visible='legendonly'), row=row, col=1, secondary_y=True)

    # Update layout
    fig.update_layout(
        template="plotly_white",
        title={
            'text': "Air Quality and Weather Data Comparison",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=1200,
        margin=dict(t=100, b=50, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title="",
            bgcolor="white",
            bordercolor="Black",
            borderwidth=2
        ),
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Air Quality Data", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Weather Data", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Air Quality Data", row=2, col=1)
    fig.update_yaxes(title_text="Weather Data", secondary_y=True, row=2, col=1)
    
    if new_data_fetched:
        update_plot_cache(cache_key, fig)
    return fig, hourly_df.index.min(), hourly_df.index.max(), df, upper_date_limit, lower_date_limit


def get_color_scale(column):
    if column == 'PM25':
        max_value = 500.4
        return [
            (0.00, "white"), (12.0 / max_value, "#0e4"),  # Good
            (12.0 / max_value, "#0e4"), (35.4 / max_value, "#ff0"),  # Moderate
            (35.4 / max_value, "#ff0"), (55.4 / max_value, "#f70"),  # Unhealthy for Sensitive Groups
            (55.4 / max_value, "#f70"), (150.4 / max_value, "#f00"),  # Unhealthy
            (150.4 / max_value, "#f00"), (250.4 / max_value, "#839"),  # Very Unhealthy
            (250.4 / max_value, "#839"), (500.4 / max_value, "#702")   # Hazardous
        ]
    elif column == 'PM10':
        max_value = 604.0
        return [
            (0.00, "white"), (54.0 / max_value, "#0e4"),  # Good
            (54.0 / max_value, "#0e4"), (154.0 / max_value, "#ff0"),  # Moderate
            (154.0 / max_value, "#ff0"), (254.0 / max_value, "#f70"),  # Unhealthy for Sensitive Groups
            (254.0 / max_value, "#f70"), (354.0 / max_value, "#f00"),  # Unhealthy
            (354.0 / max_value, "#f00"), (424.0 / max_value, "#839"),  # Very Unhealthy
            (424.0 / max_value, "#839"), (604.0 / max_value, "#702")   # Hazardous
        ]
    # else:
    #     return [
    #         (0.00, "white"), (0.33, "#999"),
    #         (0.33, "#999"), (0.66, "#444"),
    #         (0.66, "#444"), (1.00, "#000")
    #     ]
    
def calendar_heatmap(start_date=None, end_date=None, column='PM25', use_cache=True):
    cache_key = f"calendar_heatmap_{start_date}_{end_date}_{column}"
    
    if use_cache:
        cached_fig = get_cached_plot(cache_key)
        if cached_fig:
            print("Using cached plot")
            return cached_fig, None, None, False  # Assuming you want to return startDate and endDate as None for cached plots

    data_file, new_data_fetched = get_plot_data()
    daily_df = process_data(data_file, start_date, end_date, resample='D')

    daily_df = filter_valid_pm_data(daily_df)
    
    if daily_df.empty:
        print("No data available for plotting.")
        return None, None, None, new_data_fetched

    # Filter out rows where the specified column is NaN or zero
    daily_df.reset_index(inplace=True)
    daily_df = daily_df[daily_df[column].notna() & (daily_df[column] != 0) & (daily_df[column] > 0)]

    if daily_df.empty:
        print(f"No valid data available for plotting for column {column}.")
        return None, None, None, new_data_fetched

    # Create a new DataFrame with only the required columns
    daily_df = daily_df[['Datetime', column]]
    daily_df.rename(columns={'Datetime': 'ds', column: 'value'}, inplace=True)

    daily_df['day'] = daily_df['ds'].dt.day

    color_scale = get_color_scale(column)

    fig = calplot(
        daily_df,
        x="ds",
        y="value",
        years_title=True,
        text="day",  # Display the day in each cell
        gap=4,
        space_between_plots=0.15,
        colorscale=color_scale
    )
    

    if new_data_fetched:
        update_plot_cache(cache_key, fig)

    return fig, daily_df['ds'].min(), daily_df['ds'].max(), new_data_fetched



def update_t640_and_met_pickle(t640_csv, met_csv, data_file='T640_and_MET_data.pkl'):
    # Load T640 data from CSV
    t640_data = pd.read_csv(t640_csv, parse_dates=['Datetime'])
    
    # Load MET data from CSV
    met_data = pd.read_csv(met_csv, parse_dates=['Datetime'])
    
    # Load existing data from the pickle file
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)
            if isinstance(existing_data, pd.DataFrame):
                existing_df = existing_data
            else:
                existing_df = pd.DataFrame(existing_data)
    else:
        existing_df = pd.DataFrame(columns=t640_data.columns)  # Create empty DataFrame if no pickle exists

    # Concatenate new T640 and MET data with existing data
    combined_df = pd.concat([existing_df, t640_data, met_data], ignore_index=True)
    combined_df.set_index('Datetime', inplace=True)
    updated_df = combined_df.combine_first(pd.concat([t640_data, met_data]).set_index('Datetime')).reset_index()

    # Save updated data back to the same pickle file
    with open(data_file, 'wb') as f:
        pickle.dump(updated_df, f)

    print("T640_and_MET_data.pkl updated successfully.")

def create_and_resample_t640_met_pickle(t640_csv, met_csv, data_file='data/T640_and_MET_data.pkl'):
    # Load T640 data from CSV
    t640_data = pd.read_csv(t640_csv, parse_dates=['Timestamp (UTC+0)'])
    
    # Load MET data from CSV
    met_data = pd.read_csv(met_csv, parse_dates=['Timestamp (UTC+0)'])

    # Rename the Datetime (UTC+0) column to Datetime
    t640_data = t640_data.rename(columns={'Timestamp (UTC+0)': 'Datetime'})
    met_data = met_data.rename(columns={'Timestamp (UTC+0)': 'Datetime'})
    
    # Process T640 data without resampling
    t640_resampled = t640_data

    # Process MET data without resampling
    met_resampled = met_data

    # Merge T640 and MET data and sort in ascending order
    merged_data = pd.merge(t640_resampled, met_resampled, on='Datetime', how='outer').sort_values(by='Datetime')

    # Save the merged data to the pickle file
    with open(data_file, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f"{data_file} created/updated successfully with resampled data.")

# create_and_resample_t640_met_pickle("data/T640.csv", "data/MET.csv")

# Example usage
if __name__ == "__main__":
    data_file = get_plot_data()
    
    # # Save hourly data to CSV
    # save_data_to_csv(data_file, "hourly_data.csv", resample='H')
    
    # # Save daily data to CSV
    # save_data_to_csv(data_file, "daily_data.csv", resample='D')




