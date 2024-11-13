from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate  # Import Migrate
import os

if not os.path.exists('database'):
    os.makedirs('database')

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'live.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = 'e277d64e869d7ec4b1c2e26190dfc2fd'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/database.db3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Migrate

class Sensor(db.Model):
    __tablename__ = 'sensors'  # Specify the table name

    id = db.Column(db.Integer, primary_key=True)  # Sensor ID
    sensor_name = db.Column(db.String, nullable=False)  # Sensor Name
    latitude = db.Column(db.Float, nullable=False)  # Latitude
    longitude = db.Column(db.Float, nullable=False)  # Longitude
    sensor_type = db.Column(db.String, nullable=False)  # Sensor Type

    def __init__(self, sensor_name, latitude, longitude, sensor_type):
        self.sensor_name = sensor_name
        self.latitude = latitude
        self.longitude = longitude
        self.sensor_type = sensor_type

def get_sensors():
    return Sensor.query.all()

with app.app_context():
    db.create_all() 