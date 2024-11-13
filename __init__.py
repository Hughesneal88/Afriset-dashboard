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
from functions.get_t640 import get_T640
from io import BytesIO, StringIO
# from werkzeug.middleware.proxy_fix import ProxyFix
from functions.graph_plot import daily_T640, hourly_T640, save_data_to_csv, get_plot_data, process_data, create_air_quality_weather_plot
# from functions import Praxius_data2
# import time
import os
import requests
import time
from datetime import datetime
from threading import Thread
# import random
import datetime
# import os
from flask_apscheduler import APScheduler
from flask_sqlalchemy import SQLAlchemy
from functions.database import Sensor

# from Praxius_data import get_praxius_data

import hashlib
import base64
import json

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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/database.db3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
mail = Mail(app)
api_key = open("api_key", 'r').read()
app.config.from_object(Config())

# GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_MAPS_API_KEY = 'AIzaSyAeEYdpE4bUtLYAKiq1NpPW-L9FwPybwtI'

scheduler = APScheduler()
scheduler.init_app(app)

db = SQLAlchemy(app)
