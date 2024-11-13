import json
import plotly
from plotly.subplots import make_subplots
import base64
import json 
import plotly.graph_objects as go
# import tempfile
from flask import Flask, render_template, send_file, url_for, request, redirect, Response
from flask_mail import Mail, Message 
from functions.get_city import get_city_details, get_device_rec
from functions.get_t640 import get_T640, get_forecast_df, ip_info
from functions.RNNmodel import forecast_with_rnn, is_model_trained, train_model
from io import BytesIO, StringIO
# from werkzeug.middleware.proxy_fix import ProxyFix
from functions.graph_plot import create_and_resample_t640_met_pickle, daily_T640, hourly_T640, save_data_to_csv, get_plot_data, process_data, create_air_quality_weather_plot, return_df, calendar_heatmap
# from functions import Praxius_data2
# import time
import os
import requests
import time
from datetime import datetime
from threading import Thread
# import random
import datetime
import os
from flask_apscheduler import APScheduler

# from Praxius_data import get_praxius_data

import hashlib
import base64
import json
from flask import jsonify



import shutil

class Config:
    SCHEDULER_API_ENABLED = True
    SCHEDULER_EXECUTORS = {
        'default': {'type': 'threadpool', 'max_workers': 20}
    }
    SCHEDULER_JOB_DEFAULTS = {
        'coalesce': False,
        'max_instances': 3
    }

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'live.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = 'e277d64e869d7ec4b1c2e26190dfc2fd'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
api_key = open("api_key", 'r').read()
app.config.from_object(Config())


scheduler = APScheduler()
scheduler.init_app(app)

# Ensure the database directory exists
if not os.path.exists('database'):
    os.makedirs('database')

# Replace the Sensor class and database interactions with JSON file handling
class Sensor:
    def __init__(self, sensor_name, latitude, longitude, sensor_type):
        self.sensor_name = sensor_name
        self.latitude = latitude
        self.longitude = longitude
        self.sensor_type = sensor_type

# Function to load sensors from a JSON file
def load_sensors():
    try:
        with open('sensors.json', 'r') as f:
            sensors_data = json.load(f)
            return [Sensor(**sensor) for sensor in sensors_data]
    except FileNotFoundError:
        return []  # Return an empty list if the file does not exist

# Function to save sensors to a JSON file
def save_sensors(sensors):
    with open('sensors.json', 'w') as f:
        json.dump([sensor.__dict__ for sensor in sensors], f)

# Load sensors at the start


# Example of adding a new sensor
def add_sensor(sensor_name, latitude, longitude, sensor_type):    
    # Example usage
    # add_sensor("New Sensor", 12.34, 56.78, "Type A")
    sensors=load_sensors()
    new_sensor = Sensor(sensor_name, latitude, longitude, sensor_type)
    sensors.append(new_sensor)
    save_sensors(sensors)


mail = Mail(app)

predictions_buffer = []  # Temporary storage for predictions

# @scheduler.task('cron', id='get_plot_data_job', minute='*/5', misfire_grace_time=900)
# def scheduled_get_plot_data():
#     print("Running scheduled data fetch in background thread...")
#     get_plot_data()
#     return 0

# aqi_measure = "μgm-3"


def get_city_data():
    """Get data about the current AQI in various cities."""
    # TODO: Make this function more dynamic and add support for other cities/AQIs
    # url = "https://api.airvisual.com/v2/nearest_city?key=%s" %(api_key) 
    try:
        # city_data = get_city_details(addr)
        city_data = get_device_rec()
    except:
        city_data = ('N/A', 'N/A', 0, 'N/A', 'N/A', 'N/A')
    return city_data



class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range): # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)
color_coding = RangeDict({range(0,51):"Good", range(51,101):"Moderate", range(101,150):"Unhealthy for Sensitive Groups", range(151,201): "Unhealthy", range(201,301): "Very Unhealthy", range(301,501): "Hazardous"})

colors = {"Good": "#0dbf5f", "Moderate": "#FFFF00", "Unhealthy for Sensitive Groups": "#F28C28", "Unhealthy": "#D22B2B", "Very Unhealthy": "A020F0", "Hazardous": "#800000", "N/A": "#127222"}
def background_color(city_data):
    try:
        # background_color = colors[color_coding[get_city_data(addr)[2]]]
        background_color = colors[color_coding[city_data[2]]]
        print(background_color)
    except:
        background_color = colors["N/A"]
        print(background_color)

    return background_color


@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype= 'application/manifest+json')

@app.route('/sw.js')
def sw():
    return send_file('sw.js', mimetype='application/javascript')

@app.route('/regions')
def region():
    context = {'region' : 'active'}
    # data = get_city_data()
    return render_template('<h1>Coming Soon</h1>', context=context)

tipsDict = [["Check the AQI and Follow Associated Guidance", "Given that air quality can vary on a daily basis, it is advisable to check the air quality information for your area in the morning and adjust your plans accordingly. Depending on the Air Quality Index (AQI), it may be necessary to reduce your time spent outdoors and stay indoors where the air is cleaner."],
            ["Put a Hold on Any Outdoor Workouts", "It is crucial for everyone to steer clear of intense physical activity outdoors, particularly for the general public. With the ongoing increase in temperature, it's essential for individuals to recognize that engaging in strenuous activities in hot weather, especially when air quality is poor, can result in severe consequences."],
            ["Wear a Mask When You Need to Head Outside", "Wearing a properly fitted N95 mask can effectively filter out more than 95'%' of fine particulate matter, offering a way to lower your exposure to wildfire smoke when outdoors. Nevertheless, it's important to note that wildfires also emit harmful gaseous pollutants, which these masks do not filter. Therefore, it's still crucial to minimize exposure on days with hazardous air quality and only go outside when absolutely necessary.If you must be outdoors, refrain from engaging in strenuous activities and wear an N95 mask regardless of your health condition."],
            ["Wash Your Clothes (And Your Body) When You Return Home", "Ensure that you remove the clothes you wear outdoors in a designated area within your home, as the [particulate material] tends to cling to your skin, hair, and clothing, Individuals who spend extended periods of time outside due to work should wash these clothes separately from the rest of the household laundry. Additionally, it is advisable to take off your shoes at the entrance of your home to prevent the spread of particulate material throughout the house. Taking a bath or shower shortly after returning home is also recommended to eliminate any debris from your hair and skin."],
            ["Protect Your Indoor Air Quality", "Regularly inspect and clean or replace the filters in your air conditioner. Opt for HEPA filters whenever possible, as they are specifically designed to capture even the tiniest particles and dust. Additionally, ensure that the batteries in your carbon monoxide detectors are in good working condition to guarantee their effectiveness."],
            ["Stop Smoking", "Smoking significantly impacts air quality, emitting toxic pollutants into the environment. Secondhand smoke worsens indoor air quality and poses immediate health risks. Smoking also contributes to outdoor air pollution through carbon dioxide emissions. Measures to improve air quality include regularly cleaning air conditioner filters and using high-efficiency filters. It is crucial to reduce smoking and adopt practices that promote clean indoor and outdoor air."],
            ["Keep Close Tabs on Preexisting Pulmonary Conditions","Regularly monitor air quality indexes and forecasts in your area. Stay informed about air pollution levels and take necessary precautions, such as limiting outdoor activities during times of poor air quality or using air purifiers indoors. This proactive approach can help minimize exposure to pollutants and support respiratory health."]
            ]
legendval = ["0 - 50", "51-100", "101-150", "151-200", "201-300", "  301+"]
legendcol = ["#00c954", "#FFFF00", "#F28C28", "#D22B2B", "#A020F0", "#800000"]
legendextra = ["""
               Healthy
               Enjoy the fresh air! It's a great day for outdoor activities.
               """,
               """
               Moderate
               Air quality is acceptable, but sensitive individuals may experience minor discomfort.
               """,
               """
               Unhealthy for Sensitive Groups
               People with respiratory or heart conditions, children, and older adults should reduce prolonged outdoor exertion.
               """,
               """
               Unhealthy
               Everyone may begin to experience health effects; limit outdoor activities, especially if you fall into sensitive groups.
               """, 
               """
               Very Unhealthy
               Health warnings of emergency conditions. The entire population is likely to be affected. Stay indoors if possible.
               """, 
               """
               Hazardous
               Health alert: everyone may experience more serious health effects. Avoid outdoor activities, and remain indoors if possible.
               """
               ]
info = [["Healthy ","Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
          ["Air quality is considered satisfactory, and air pollution poses little or no risk", "Air quality is acceptable, however, there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.",
            "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
            "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects",
            "Health warnings of emergency conditions. The entire population is more likely to be affected.",
            "Health alert: everyone may experience more serious health effects"
            ], 
          ["It's a good time to go outside", 
           "People unusually sensitive to air pollution, such as the asthmatic, should limit prolonged outdoor exertion and wear a mask.",
            "Young children, the aged, and people unusually sensitive to air pollution, such as the asthmatic, should <b>limit<b> prolonged outdoor exertion and wear a mask.",
            "Young children, the aged, and people unusually sensitive to air pollution, such as the asthmatic, should <b>avoid<b> prolonged outdoor exertion; everyone else, especially children, should <b>limit<b> prolonged outdoor exertion and wear a mask.",
            "Active children and adults, and people with respiratory disease, such as asthma, should <b>avoid<b> all outdoor exertion; everyone else, especially children, should <b>limit<b> outdoor exertion and wear a mask.",
            "Everyone should <b>avoid<b> all outdoor exertion, wear a mask, and <b>remain<b> indoors if possible."
            ]]

def gettips():
    # tip  = random.choice(tipsDict)
    return "legend coming soon..."

# @app.route('/get-ip', methods=['GET'])
# def get_ip():
#     ip = request.access_route[0]
#     return ip

# @app.route('/pm25')
# def pm1():
#     context = {'pm25' : 'active'}
#     addr = str(request.access_route[0])
#     data = get_city_data(addr)
#     return render_template('index.html', context=context, city=data, color=background_color(data), warning=color_coding[data[2]])


def get_aqi_description(aqi_value):
    if aqi_value <= 50:
        return [info[0][0],info[1][0],info[2][0], legendval[0]]
    elif aqi_value <= 100:
        return [info[0][1],info[1][1],info[2][1], legendval[1]]
    elif aqi_value <= 150:
        return [info[0][2],info[1][2],info[2][2], legendval[2]] 
    elif aqi_value <= 200:
        return [info[0][3],info[1][3],info[2][3], legendval[3]]
    elif aqi_value <= 300:
        return [info[0][4],info[1][4],info[2][4], legendval[4]]
    else:
        return [info[0][5],info[1][5],info[2][5], legendval[5]]


# Function to fetch and store forecast data in the background
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
            
            forecast_values = forecast_with_rnn(df)  # Calculate the forecast using the RNN
            predicted_value = forecast_values['PM25'].iloc[-1]  # Get the last predicted value
            predictions_buffer.append(predicted_value)  # Store the prediction
            
            # Optionally, save the forecast values
            save_forecast(forecast_values)
        except Exception as e:
            print(f"Error during forecasting: {e}")
        
        time.sleep(3600)  # Update every hour

# Function to save forecast data (implement this as needed)
def save_forecast(forecast_values):
    with open('forecast.json', 'w') as f:
        json.dump(forecast_values.to_dict(), f)

# Function to load forecast data
def load_forecast():
    try:
        with open('forecast.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Return an empty dict if the file does not exist

# Start the forecast updater in a separate thread
forecast_thread = Thread(target=update_forecast)
forecast_thread.daemon = True
forecast_thread.start()

@app.route('/')
def index():
    context = {'home' : 'active'}
    addr = str(request.access_route[0])  
    city, country = ip_info(addr) if ip_info(addr) else ('Accra','Ghana')
    data = get_T640(city, country) if get_T640(city, country) else ('N/A', 'N/A', 0, 'N/A', 'N/A', 'N/A')
    data[3] = data[3].strftime("%a, %d %B %Y %I:%M %p")
    
    # Load the latest forecast values
    forecast_values = load_forecast()
    print(forecast_values)
    print('City and country:', city, country)
    
    return render_template('index.html', 
                           name='Home', 
                           context=context, 
                           data=data, 
                           city=city,
                           country=country,
                           region=region,
                           color=background_color(data),
                           warning=color_coding[data[2]], 
                           tips=gettips(), 
                           forecast=forecast_values,
                           legendval=legendval, 
                           legendcol=legendcol, 
                           legendextra=legendextra, 
                           aqi_description=get_aqi_description(data[2])
                           )

def obfuscate_data(data):
    json_string = json.dumps(data)
    sha256 = hashlib.sha256()
    sha256.update(json_string.encode('utf-8'))
    hash_prefix = sha256.hexdigest()[:16]  # Use first 16 characters of hash as a prefix
    obfuscated = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    return f"{hash_prefix}:{obfuscated}"

@app.route('/graph/<graphtype>', methods=['POST', 'GET'])
def plot_graph(graphtype):
    # Set default dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    start_date_hourly = end_date - datetime.timedelta(days=7)
    start_date = start_date.strftime('%Y-%m-%dT%H:%M')
    start_date_hourly = start_date_hourly.strftime('%Y-%m-%dT%H:%M')
    end_date = end_date.strftime('%Y-%m-%dT%H:%M')

    # Get dates from form if provided
    if request.method == 'POST':
        form_start_date = request.form.get('start_date')
        form_end_date = request.form.get('end_date')
        
        # Update dates if provided in the form
        if form_start_date:
            start_date = form_start_date
            start_date_hourly = form_start_date
        if form_end_date:
            end_date = form_end_date

    context = {'graph': 'active'}

    # Convert string dates to datetime objects for the graph functions
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M')
    start_hourly_datetime = datetime.datetime.strptime(start_date_hourly, '%Y-%m-%dT%H:%M')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M')

    if graphtype == "daily":
        view_start_date=start_datetime.strftime('%a, %d %B %Y %I:%M %p')
        view_end_date=end_datetime.strftime('%a, %d %B %Y %I:%M %p')
        fig, min_date, max_date, new_data_fetched = daily_T640(start_datetime, end_datetime)
    elif graphtype == "hourly":
        start_datetime = start_hourly_datetime
        view_start_date=start_datetime.strftime('%a, %d %B %Y %I:%M %p')
        view_end_date=end_datetime.strftime('%a, %d %B %Y %I:%M %p')
        fig, min_date, max_date, new_data_fetched = hourly_T640(start_datetime, end_datetime)
    elif graphtype == "aq_weather":
        view_start_date=start_datetime.strftime('%a, %d %B %Y %I:%M %p')
        view_end_date=end_datetime.strftime('%a, %d %B %Y %I:%M %p')
        fig, min_date, max_date, new_data_fetched, upper_date_limit, lower_date_limit = create_air_quality_weather_plot(start_datetime, end_datetime)
    # elif graphtype == "calendar":
    #     view_start_date = start_datetime.strftime('%a, %d %B %Y %I:%M %p')
    #     view_end_date = end_datetime.strftime('%a, %d %B %Y %I:%M %p')
    #     column = request.form.get('column', 'PM25')  # Default to 'PM25' if no column is specified
    #     fig, min_date, max_date, new_data_fetched = calendar_heatmap(start_date=start_datetime, end_date=end_datetime, column=column)
    else:
        return "Invalid graph type", 400

    graph_json = fig.to_json()
    obfuscated_json = obfuscate_data(json.loads(graph_json))

    return render_template('graph.html', 
                           name="Graph Plots", 
                           context=context,
                           graph_type=graphtype,
                           graph_json=obfuscated_json, 
                           start_date=min_date,
                           end_date=max_date,
                           view_start_date=view_start_date,
                           view_end_date=view_end_date,
                           new_data_fetched=new_data_fetched,
                           )

@app.route('/graph/<graph_type>/download', methods=['POST'])
def download_file(graph_type):
    download_type = request.form.get('download_type')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    # Parse start_date and end_date if they're provided
    if start_date:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    if end_date:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    if graph_type == "daily":
        resample = 'D'
        fig, min_date, max_date, new_data_fetched, upper_date_limit, lower_date_limit = daily_T640(start_date, end_date)
    elif graph_type == "hourly":
        resample = 'H'
        fig, min_date, max_date, new_data_fetched, upper_date_limit, lower_date_limit = hourly_T640(start_date, end_date)
    elif graph_type == "aq_weather":
        fig, min_date, max_date, df, upper_date_limit, lower_date_limit = create_air_quality_weather_plot(start_date, end_date)
    else:
        return "Invalid graph type", 400

    if download_type == 'png':
        img_bytes = fig.to_image(format="png")
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/png',
            download_name=f'{graph_type}_graph_{start_date}_{end_date}.png',
            as_attachment=True
        )
    elif download_type == 'svg':
        img_bytes = fig.to_image(format="svg")
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/svg+xml',
            download_name=f'{graph_type}_graph_{start_date}_{end_date}.svg',
            as_attachment=True
        )
    elif download_type == 'jpeg':
        img_bytes = fig.to_image(format="jpeg")
        return send_file(
            BytesIO(img_bytes),
            mimetype='image/jpeg',
            download_name=f'{graph_type}_graph_{start_date}_{end_date}.jpeg',
            as_attachment=True
        )
    else:
        return "Invalid download type", 400
    """
    # Commented out CSV download functionality
    elif download_type == 'csv':
        if graph_type in ["daily", "hourly"]:
            plot_data = get_plot_data()
            csv_buffer = save_data_to_csv(plot_data, start_date, end_date, resample)
        else:  # aq_weather
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=True)
            csv_buffer.seek(0)
        
        return send_file(
            BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            download_name=f'{graph_type}_data_{start_date}_{end_date}.csv',
            as_attachment=True
        )
    """

@app.route('/contact-us', methods=['POST', 'GET'])
def contact_us():
    context = {'contact' : 'active'}
    if request.method == 'POST':
        email = request.form.get('email-address')
        messagebody = request.form.get('messageBody')
        msg = Message(subject=f'feedback from {email}', sender='mailtrap@demomailtrap.com', recipients=["hughesneal88@gmail.com"])
        msg.body = messagebody
        mail.send(msg)
        return redirect(url_for('index'))
    return render_template('contactus.html')

@app.route('/graph/calendar', methods=['POST', 'GET'])
def calendar_plot():
    # Set default dates
    end_date = datetime.datetime.now()
    start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.strftime('%Y-%m-%dT%H:%M')
    end_date = end_date.strftime('%Y-%m-%dT%H:%M')

    # Get dates from form if provided
    if request.method == 'POST':
        form_start_date = request.form.get('start_date')
        form_end_date = request.form.get('end_date')

        # Update dates if provided in the form
        if form_start_date:
            start_date = form_start_date
        if form_end_date:
            end_date = form_end_date

    context = {'graph': 'active'}
    # Convert string dates to datetime objects for the graph functions
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M')

    view_start_date = start_datetime.strftime('%a, %d %B %Y %I:%M %p')
    view_end_date = end_datetime.strftime('%a, %d %B %Y %I:%M %p')
    column = request.form.get('column', 'PM25')  # Default to 'PM25' if no column is specified
    fig, min_date, max_date, new_data_fetched = calendar_heatmap(start_date=start_datetime, end_date=end_datetime, column=column)

    if fig is None:
        return "No data available for the specified date range", 400

    graph_json = fig.to_json()
    obfuscated_json = obfuscate_data(json.loads(graph_json))

    return render_template('graph.html',
                           name="Calendar Plot",
                           context=context,
                           graph_type="calendar",
                           graph_json=obfuscated_json,
                           start_date=min_date,
                           end_date=max_date,
                           view_start_date=view_start_date,
                           view_end_date=view_end_date,
                           new_data_fetched=new_data_fetched)


@app.route('/graph/aq_weather', methods=['GET', 'POST'])
def air_quality_weather():
    # end_date = request.args.get('end_date')
    # start_date = request.args.get('start_date')
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    start_date = start_date.strftime('%Y-%m-%dT%H:%M')
    end_date = end_date.strftime('%Y-%m-%dT%H:%M')
    # download_type = request.args.get('download')
    selected_columns = request.args.get('selected_columns', '').split(',')
    selected_columns = [col.strip() for col in selected_columns if col.strip()]  # Remove empty strings and whitespace



    if request.method == 'POST':
        form_start_date = request.form.get('start_date')
        form_end_date = request.form.get('end_date')
        
        # Update dates if provided in the form
        if form_start_date:
            start_date = form_start_date
        if form_end_date:
            end_date = form_end_date

    # Convert string dates to datetime objects for the graph functions
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M')
    fig, start_date, end_date, df, upper_date_limit, lower_date_limit = create_air_quality_weather_plot(start_datetime, end_datetime)
    
    # if download_type == 'png':
    #     img_bytes = fig.to_image(format="png")
    #     img_io = BytesIO(img_bytes)
    #     img_io.seek(0)
    #     return send_file(img_io, mimetype='image/png', download_name='aq_weather_plot.png')
    # elif download_type == 'csv':
    #     if not selected_columns:
    #         selected_columns = df.columns.tolist()  # Use all columns if none selected
        
    #     # Ensure all selected columns exist in the dataframe
    #     valid_columns = [col for col in selected_columns if col in df.columns]
        
    #     if not valid_columns:
    #         return "No valid columns selected", 400

    #     csv_data = df[valid_columns].to_csv(index=True)
    #     csv_buffer = StringIO(csv_data)
    #     csv_buffer.seek(0)
    #     return send_file(
    #         csv_buffer,
    #         mimetype="text/csv",
    #         download_name=f"aq_weather_data_{start_date}_{end_date}.csv",
    #         as_attachment=True
    #     )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    obfuscated_json = obfuscate_data(json.loads(graphJSON))
    return render_template('graph.html', 
                           graph_json=obfuscated_json, 
                           start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                           view_start_date=start_date.strftime('%a, %d %B %Y %H:%M:%S'),
                           view_end_date=end_date.strftime('%a, %d %B %Y %H:%M:%S'),
                           end_date=end_date.strftime('%Y-%m-%d %H:%M:%S'), 
                           title="Air Quality and Weather Comparison")


# @app.route('/map')
# def display_map():
#     sensors = load_sensors()
#     print(sensors[0].sensor_name)
#     sensor_data = [{"name": sensor.sensor_name, "latitude": sensor.latitude, "longitude": sensor.longitude} for sensor in sensors]
#     return render_template('map.html', sensors=sensor_data)

    # print(GOOGLE_MAPS_API_KEY)
    # return render_template('map.html', 
    #                        api_key=GOOGLE_MAPS_API_KEY, 
    #                        sensor=SENSOR_LOCATION)

# class Config:
#     JOBS = [
#         {
#             'id': 'job1',
#             'func': job1,
#             'args': (),
#             'trigger': 'interval',
#             'minutes': 30,
#         },
#         {
#             'id': 'job2',
#             'func': job2,
#             'args': (),
#             'trigger': 'interval',
#             'minutes': 30,
#         }
#     ]

#     SCHEDULER_API_ENABLED = True


# Reloader Function
def reload_website():    
    url = "https://afriset-air-monitor.onrender.com/"  # Replace with your Render URL
    interval = 60  # Interval in seconds
    while True:
        try:
            response = requests.get(url)
            print(f"Reloaded at {datetime.datetime.now().isoformat()}: Status Code {response.status_code}")
        except requests.RequestException as error:
            print(f"Error reloading at {datetime.datetime.now().isoformat()}: {error}")
        time.sleep(interval)

# Start the reloader in a separate thread
reloader_thread = Thread(target=reload_website)
reloader_thread.daemon = True  # This allows the thread to exit when the main program does
# reloader_thread.start()

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to create and resample T640 and MET data
@app.route('/upload-missing-data', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the files
        if 't640_file' not in request.files or 'met_file' not in request.files:
            return "No file part", 400
        
        t640_file = request.files['t640_file']
        met_file = request.files['met_file']

        if t640_file.filename == '' or met_file.filename == '':
            return "No selected file", 400

        # Save the uploaded files
        t640_path = os.path.join(UPLOAD_FOLDER, t640_file.filename)
        met_path = os.path.join(UPLOAD_FOLDER, met_file.filename)
        t640_file.save(t640_path)
        met_file.save(met_path)

        # Backup the existing T640_and_MET_data.pkl if it exists
        if os.path.exists('data/T640_and_MET_data.pkl'):
            shutil.move('data/T640_and_MET_data.pkl', 'data/T640_and_MET_data_backup.pkl')

        # Create a new T640_and_MET_data.pkl
        create_and_resample_t640_met_pickle(t640_path, met_path)

        return redirect(url_for('upload_file'))

    return render_template('upload.html')

@app.route('/upload')
def upload_csv():
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    # app.config.from_object(Config())

    # scheduler = APScheduler()
    # # it is also possible to enable the API directly
    # # scheduler.api_enabled = True
    # scheduler.init_app(app)
    scheduler.start()
    
    app.run('0.0.0.0', 5000)

    # app.run(host='192.168.137.1', port=5000, debug=True)