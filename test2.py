from flask import Flask, request, jsonify, render_template, render_template, Blueprint
from pymongo import MongoClient
from dotenv import load_dotenv
import requests
import os
from werkzeug.security import generate_password_hash, check_password_hash
import re
from flask_cors import CORS
import pickle
import numpy as np
import matplotlib.pylab as plt
from skimage.transform import resize
from datetime import datetime
from PIL import Image
import json
from io import BytesIO
from plantheight import calculate_plant_height
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import tempfile
import ee
import google.generativeai as genai
import os
from datetime import datetime, timedelta
from utils.captcha import verify_captcha
from ee_init import initialize_earth_engine

load_dotenv()

bp1 = Blueprint('bp1', __name__)

# Global variable to track Earth Engine initialization status
ee_initialized = False

cloudinary.config(
  cloud_name = os.getenv("CLOUD_NAME"),
  api_key = os.getenv("CLOUD_API_KEY"),
  api_secret = os.getenv("CLOUD_API_SECRET")
)

disease_info = pd.read_csv('datasets/disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('datasets/supplement_info.csv',encoding='cp1252')
class TempModel(nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))

    def forward(self, inp):
        return self.conv1(inp)

# # Initialize Google Earth Engine
# try:
#     ee.Initialize(project='agronexus-457107')
# except Exception as e:
#     print("Please authenticate Google Earth Engine first")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def get_ndvi_ndwi(latitude, longitude):
    """Calculate NDVI and NDWI for the given location"""
    if not ee_initialized:
        raise RuntimeError("Earth Engine not initialized")
        
    point = ee.Geometry.Point([longitude, latitude])
    
    # Get the date range (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Get Sentinel-2 imagery
    sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(point) \
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()
    
    # Calculate NDVI
    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Calculate NDWI
    ndwi = sentinel.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # Get the values at the point
    ndvi_value = ndvi.reduceRegion(ee.Reducer.mean(), point, 10).get('NDVI').getInfo()
    ndwi_value = ndwi.reduceRegion(ee.Reducer.mean(), point, 10).get('NDWI').getInfo()
    
    return {
        'ndvi': ndvi_value,
        'ndwi': ndwi_value
    }

def get_soil_type(latitude, longitude):
    """Get soil type information using Google Earth Engine"""
    if not ee_initialized:
        raise RuntimeError("Earth Engine not initialized")
        
    point = ee.Geometry.Point([longitude, latitude])
    
    # Use OpenLandMap soil pH dataset
    soil = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
    soil_value = soil.reduceRegion(ee.Reducer.mean(), point, 10).get('b0').getInfo()
    
    # Convert the pH value to a more readable format (original values are in pH * 10)
    if soil_value is not None:
        soil_value = soil_value / 10.0
    else:
        soil_value = 7.0  # Default neutral pH if no data available
    
    return soil_value

def get_weather_data(latitude, longitude):
    """Get historical weather data"""
    if not ee_initialized:
        raise RuntimeError("Earth Engine not initialized")
        
    point = ee.Geometry.Point([longitude, latitude])
    
    # Get ERA5 monthly data
    weather = ee.ImageCollection('ECMWF/ERA5/MONTHLY') \
        .filterBounds(point) \
        .select(['total_precipitation', 'temperature_2m']) \
        .mean()
    
    weather_data = weather.reduceRegion(ee.Reducer.mean(), point, 10).getInfo()
    
    return weather_data

# def get_crop_recommendation(ndvi, ndwi, soil_ph, weather_data, farmer_responses):
#     """Generate crop recommendation using Gemini API"""
#     # Convert USD to INR (approximate conversion)
#     budget_inr = float(farmer_responses.get('budget', 0)) * 83

#     # Get the current season based on month
#     current_month = datetime.now().month
#     if 6 <= current_month <= 9:
#         season = "Kharif"
#     elif 10 <= current_month <= 2:
#         season = "Rabi"
#     else:
#         season = "Zaid"

#     prompt = f"""
#     Based on:
#     • Environment: NDVI={ndvi}, NDWI={ndwi}, pH={soil_ph}, Temp={weather_data.get('temperature_2m', 'N/A')}°C
#     • Rainfall: {weather_data.get('total_precipitation', 'N/A')} mm
#     • Season: {season}
#     • Farmer: {farmer_responses.get('experience')} years experience, {farmer_responses.get('landSize')} acres
#     • Irrigation: {farmer_responses.get('irrigation')}
#     • Past Crop: {farmer_responses.get('past_crop')}
#     • Budget: ₹{budget_inr:,.2f}
#     • Market: {farmer_responses.get('market')}

#     Provide exactly 3 crop recommendations in this format ONLY:

#     1. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]
#     2. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]
#     3. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]

#     Keep it extremely concise, one line per crop only.
#     """
    
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         print(f"Error generating recommendation: {str(e)}")
#         return "Unable to generate crop recommendations at the moment. Please try again later."

#Importing models
def load_model_safely(path, default_return=None):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load model from {path}: {str(e)}")
        return default_return

# Load models with error handling
model1 = load_model_safely('models/wheat_price_prediction.pkl')  # Wheat
model2 = load_model_safely('models/Cotton_price_prediction.pkl')  # Cotton
model3 = load_model_safely('models/Gram_price_prediction.pkl')   # Gram
model4 = load_model_safely('models/Jute_price_prediction.pkl')   # Jute
model5 = load_model_safely('models/Maize_price_prediction.pkl')  # Maize
model6 = load_model_safely('models/Moong_price_prediction.pkl')  # Moong
model7 = load_model_safely('models/Crop_Recommendation.pkl')     # Crop Recommendation

# Image size used during training
IMAGE_SIZE = (224, 224)

# Load environment variables
load_dotenv()


# app = Flask(__name__)
# CORS(app)

# Custom deserialization workaround


# MongoDB connection
client = MongoClient(os.getenv('MONGO_URI'))
db = client['AgroNexus']
users_collection = db['users']

model = models.resnet50(pretrained=False)
# Modify the final layer to match 39 output classes
model.fc = nn.Linear(model.fc.in_features, 39)
model.load_state_dict(torch.load("models/trained_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img_path):
    try:
        img = Image.open(img_path)
        img = transform(img)

        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)

        return predicted_class.item()

    except Exception as e:
        return str(e)


# Page Routes
@bp1.route('/')
def landing():
    return render_template('landingPage.html')

@bp1.route('/signup')
def signup_page():
    return render_template('signup.html')

@bp1.route('/login')
def login_page():
    return render_template('login.html')

@bp1.route('/home')
def home_page():
    return render_template('home.html')
# API Routes
@bp1.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        # Verify CAPTCHA
        if not verify_captcha(data.get('captcha')):
            return jsonify({'error': 'Invalid CAPTCHA verification'}), 400
        
        # Validate required fields
        required_fields = ['firstName', 'city', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate email format
        if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if email already exists
        if users_collection.find_one({'email': data['email']}):
            return jsonify({'error': 'Email already registered'}), 400
        
        # Validate password length
        if len(data['password']) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create user document
        user = {
            'firstName': data['firstName'],
            'city': data['city'],
            'email': data['email'],
            'password': hashed_password,
            'newsletter': data.get('newsletter', False)
        }
        
        # Insert user into database
        users_collection.insert_one(user)
        
        return jsonify({'message': 'User created successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp1.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Verify CAPTCHA
        if not verify_captcha(data.get('captcha')):
            return jsonify({'error': 'Invalid CAPTCHA verification'}), 400
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user by email
        user = users_collection.find_one({'email': data['email']})
        
        # Check if user exists and password is correct
        if user and check_password_hash(user['password'], data['password']):
            # Don't send password in response
            user_data = {
                'firstName': user['firstName'],
                'email': user['email']
            }
            return jsonify({'message': 'Login successful', 'user': user_data}), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp1.route('/predict_price', methods=['GET', 'POST'])
def predict_price():
    return render_template('price.html')

# @bp1.route('/crop_rec', methods=['GET','POST'])
# def crop_rec():
#     crop_mbp1ing = {
#     0: 'bp1le',
#     1: 'banana',
#     2: 'blackgram',
#     3: 'chickpea',
#     4: 'coconut',
#     5: 'coffee',
#     6: 'cotton',
#     7: 'grapes',
#     8: 'jute',
#     9: 'kidneybeans',
#     10: 'lentil',
#     11: 'maize',
#     12: 'mango',
#     13: 'mothbeans',
#     14: 'mungbean',
#     15: 'muskmelon',
#     16: 'orange',
#     17: 'papaya',
#     18: 'pigeonpeas',
#     19: 'pomegranate',
#     20: 'rice',
#     21: 'watermelon'
# }

#     crop = None
#     if request.method=='POST':
#         n = request.form['nitrogen']
#         p = request.form['phosphorus']
#         k = request.form['potassium']
#         t = request.form['temperature']
#         h = request.form['humidity']
#         ph = request.form['ph']
#         r = request.form['rainfall']
#         features = np.array([[n, p, k, t, h, ph, r]])
#         crop = crop_mbp1ing[model7.predict(features)[0]]
#         return render_template('crop_recom.html', crop=crop)
#     return render_template('crop_recom.html', crop=crop)

# @bp1.route('/disease_detect', methods=['POST','GET'])
# def disease_detect():
#     predicted_label=None
#     confidence=None
#     if request.method=='POST':
#         file = request.files['image']
#         if file:
#             # Read image in-memory
#             img = Image.open(BytesIO(file.read())).convert("RGB")
#             img = img.resize(IMAGE_SIZE)
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0) / 255.0

#             prediction = model.predict(img_array)[0]
#             predicted_label = labels[np.argmax(prediction)]
#             confidence = round(100 * np.max(prediction), 2)
#             print(prediction, predicted_label, confidence)
#             return render_template('disease.html', disease=predicted_label, accuracy_score=confidence)
#     return render_template('disease.html', disease=predicted_label, accuracy_score=confidence)

# @bp1.route('/disease_detect', methods=['POST', 'GET'])
# def disease_detect():
#     predicted_label = None
#     confidence = None
#     image_url = None

#     if request.method == 'POST':
#         file = request.files['image']
#         if file:
#             # ✅ Upload to Cloudinary
#             upload_result = cloudinary.uploader.upload(file)
#             image_url = upload_result['secure_url']

#             # ✅ Download from Cloudinary and prepare image for model
#             response = requests.get(image_url)
#             img = Image.open(BytesIO(response.content)).convert("RGB")
#             img = img.resize(IMAGE_SIZE)
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0) / 255.0

#             # ✅ Predict
#             prediction = model.predict(img_array)[0]
#             predicted_label = labels[np.argmax(prediction)]
#             confidence = round(100 * np.max(prediction), 2)

#             return render_template('disease.html',
#                                    disease=predicted_label,
#                                    accuracy_score=confidence,
#                                    image_url=image_url)

#     return render_template('disease.html',
#                            disease=predicted_label,
#                            accuracy_score=confidence,
#                            image_url=image_url)


@bp1.route('/height', methods=['GET', 'POST'])
def height():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return "No file uploaded"

        result = calculate_plant_height(file)
        if "error" in result:
            return result["error"]

        return render_template('height.html', height=result["height"], image=result["image_base64"])
    return render_template('height.html')

# @bp1.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'POST':
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static','uploads', filename)
#         image.save(file_path)
#         print(file_path)
#         pred = predict(file_path)  # Call the predict function here
#         title = disease_info['disease_name'][pred]
#         description = disease_info['description'][pred]
#         prevent = disease_info['Possible Steps'][pred]
#         image_url = disease_info['image_url'][pred]
#         supplement_name = supplement_info['supplement name'][pred]
#         supplement_image_url = supplement_info['supplement image'][pred]
#         supplement_buy_link = supplement_info['buy link'][pred]
#         return render_template('disease.html', title=title, desc=description, prevent=prevent,
#                                image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)
#     return render_template('disease.html')

@bp1.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        file = request.files['image']

        # ✅ Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(file)
        image_url_uploaded = upload_result['secure_url']  # Get Cloud URL

        # ✅ Download image back from Cloudinary
        response = requests.get(image_url_uploaded)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # ✅ Save temporarily to use predict(img_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name  # Path to pass to predict()

        # ✅ Run prediction
        pred = predict(temp_path)

        # ✅ Clean up temporary file (optional, but good practice)
        os.remove(temp_path)

        # Fetch disease/supplement info
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        disease_image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('disease.html',
                               title=title,
                               desc=description,
                               prevent=prevent,
                               image_url=disease_image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image_url,
                               buy_link=supplement_buy_link,
                               uploaded_img_url=image_url_uploaded)

    return render_template('disease.html')

@bp1.route('/problemsSolved')
def problemsSolved():
    return render_template('problems.html')

# @bp1.route('/crop_rec')
# def index():
#     return render_template('index.html')

# @bp1.route('/get_recommendation', methods=['POST'])
# def get_recommendation():
#     try:
#         data = request.json
#         latitude = data.get('latitude')
#         longitude = data.get('longitude')
#         farmer_responses = data.get('responses', {})
        
#         print(f"Processing request for location: {latitude}, {longitude}")
        
#         # Get environmental data
#         try:
#             vegetation_data = get_ndvi_ndwi(latitude, longitude)
#             print(f"Vegetation data: {vegetation_data}")
#         except Exception as e:
#             print(f"Error getting vegetation data: {str(e)}")
#             vegetation_data = {'ndvi': 0.5, 'ndwi': 0.3}  # Default values
            
#         try:
#             soil_data = get_soil_type(latitude, longitude)
#             print(f"Soil pH: {soil_data}")
#         except Exception as e:
#             print(f"Error getting soil data: {str(e)}")
#             soil_data = 7.0  # Neutral pH as default
            
#         try:
#             weather_data = get_weather_data(latitude, longitude)
#             print(f"Weather data: {weather_data}")
#         except Exception as e:
#             print(f"Error getting weather data: {str(e)}")
#             weather_data = {
#                 'temperature_2m': 25,
#                 'total_precipitation': 100
#             }  # Default values
        
#         # Get crop recommendation
#         recommendation = get_crop_recommendation(
#             vegetation_data['ndvi'],
#             vegetation_data['ndwi'],
#             soil_data,
#             weather_data,
#             farmer_responses
#         )
        
#         return jsonify({
#             'status': 'success',
#             'recommendation': recommendation,
#             'environmental_data': {
#                 'ndvi': vegetation_data['ndvi'],
#                 'ndwi': vegetation_data['ndwi'],
#                 'soil_ph': soil_data,
#                 'weather': weather_data
#             }
#         })
        
#     except Exception as e:
#         print(f"Error in get_recommendation: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         })

@bp1.before_app_first_request
def setup():
    """Initialize Earth Engine before the first request"""
    global ee_initialized
    try:
        initialize_earth_engine()
        ee_initialized = True
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Earth Engine: {str(e)}")
        ee_initialized = False

@bp1.route('/get_ndvi', methods=['POST'])
def get_ndvi():
    if not ee_initialized:
        return jsonify({
            'error': 'Earth Engine not initialized. Please try again later.'
        }), 503
        
    try:
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        
        indices = get_ndvi_ndwi(latitude, longitude)
        return jsonify(indices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp1.route('/get_weather', methods=['POST'])
def get_weather():
    if not ee_initialized:
        return jsonify({
            'error': 'Earth Engine not initialized. Please try again later.'
        }), 503
        
    try:
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        
        weather_data = get_weather_data(latitude, longitude)
        soil_ph = get_soil_type(latitude, longitude)
        
        return jsonify({
            'weather': weather_data,
            'soil_ph': soil_ph
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
