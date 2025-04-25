from flask import Flask, request, jsonify, render_template, Blueprint
import ee
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import requests

bp2 = Blueprint('bp2', __name__)

# Initialize Google Earth Engine
try:
    # Try to use service account credentials if available
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials = ee.ServiceAccountCredentials(
            email=os.getenv('EE_SERVICE_ACCOUNT'),
            key_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        ee.Initialize(credentials, project='agronexus-457107')
    else:
        # Fallback to default initialization
        ee.Initialize(project='agronexus-457107')
except Exception as e:
    print(f"Earth Engine initialization error: {str(e)}")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')  # Using gemini-pro instead of gemini-2.0-flash
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    model = None

def get_ndvi_ndwi(latitude, longitude):
    """Calculate NDVI and NDWI for the given location"""
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
    point = ee.Geometry.Point([longitude, latitude])
    
    # Use OpenLandMap soil pH dataset instead
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
    point = ee.Geometry.Point([longitude, latitude])
    
    # Get ERA5 monthly data
    weather = ee.ImageCollection('ECMWF/ERA5/MONTHLY') \
        .filterBounds(point) \
        .select(['total_precipitation', 'temperature_2m']) \
        .mean()
    
    weather_data = weather.reduceRegion(ee.Reducer.mean(), point, 10).getInfo()
    
    return weather_data

def get_crop_recommendation(ndvi, ndwi, soil_ph, weather_data, farmer_responses):
    """Generate crop recommendation using Gemini API via direct HTTP request."""
    # Convert USD to INR
    budget_inr = float(farmer_responses.get('budget', 0)) * 83

    # Get current season
    current_month = datetime.now().month
    if 6 <= current_month <= 9:
        season = "Kharif"
    elif 10 <= current_month <= 2:
        season = "Rabi"
    else:
        season = "Zaid"

    prompt = f"""
    Based on:
    • Environment: NDVI={ndvi}, NDWI={ndwi}, pH={soil_ph}, Temp={weather_data.get('temperature_2m', 'N/A')}°C
    • Rainfall: {weather_data.get('total_precipitation', 'N/A')} mm
    • Season: {season}
    • Farmer: {farmer_responses.get('experience')} years experience, {farmer_responses.get('landSize')} acres
    • Irrigation: {farmer_responses.get('irrigation')}
    • Past Crop: {farmer_responses.get('past_crop')}
    • Budget: ₹{budget_inr:,.2f}
    • Market: {farmer_responses.get('market')}

    Provide exactly 3 crop recommendations in this format ONLY:

    1. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]
    2. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]
    3. [Crop Name] - [Expected yield/acre] - [Current market rate/quintal]

    Keep it extremely concise, one line per crop only.
    """

    headers = {"Content-Type": "application/json"}
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    api_key = os.getenv("GOOGLE_API_KEY")
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, params={"key": api_key})
        print(f"Gemini API response status: {response.status_code}")
        print(f"Gemini API raw response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            # Try-safe extraction
            candidates = result.get("candidates")
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                return candidates[0]["content"]["parts"][0].get("text", "No text found.")
            else:
                return "Received an unexpected response format from Gemini API."
        else:
            return f"Gemini API error: {response.status_code} - {response.text}"

    except Exception as e:
        print(f"Exception while calling Gemini API: {str(e)}")
        return "Unable to generate crop recommendations at the moment. Please try again later."
@bp2.route('/track')
def track():
    return render_template('track.html')

@bp2.route('/crop_rec')
def index():
    return render_template('index.html')

@bp2.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        data = request.json
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        farmer_responses = data.get('responses', {})
        
        print(f"Processing request for location: {latitude}, {longitude}")
        
        # Get environmental data
        try:
            vegetation_data = get_ndvi_ndwi(latitude, longitude)
            print(f"Vegetation data: {vegetation_data}")
        except Exception as e:
            print(f"Error getting vegetation data: {str(e)}")
            vegetation_data = {'ndvi': 0.5, 'ndwi': 0.3}  # Default values
            
        try:
            soil_data = get_soil_type(latitude, longitude)
            print(f"Soil pH: {soil_data}")
        except Exception as e:
            print(f"Error getting soil data: {str(e)}")
            soil_data = 7.0  # Neutral pH as default
            
        try:
            weather_data = get_weather_data(latitude, longitude)
            print(f"Weather data: {weather_data}")
        except Exception as e:
            print(f"Error getting weather data: {str(e)}")
            weather_data = {
                'temperature_2m': 25,
                'total_precipitation': 100
            }  # Default values
        
        # Get crop recommendation
        recommendation = get_crop_recommendation(
            vegetation_data['ndvi'],
            vegetation_data['ndwi'],
            soil_data,
            weather_data,
            farmer_responses
        )
        
        return jsonify({
            'status': 'success',
            'recommendation': recommendation,
            'environmental_data': {
                'ndvi': vegetation_data['ndvi'],
                'ndwi': vegetation_data['ndwi'],
                'soil_ph': soil_data,
                'weather': weather_data
            }
        })
        
    except Exception as e:
        print(f"Error in get_recommendation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
