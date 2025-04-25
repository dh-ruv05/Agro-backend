from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import json
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyChpIrLMzJc42ETm0jS4KiKC_ra9Gv1_vE"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# WeatherAPI.com configuration
WEATHER_API_KEY = "5c4e1379ec8847fea4e101640251704"  # Your WeatherAPI.com key
WEATHER_API_BASE_URL = "http://api.weatherapi.com/v1/current.json"

# Mock data for testing (fallback)
MOCK_WEATHER_DATA = {
    "temperature": 25.5,
    "humidity": 65,
    "wind_speed": 3.2,
    "rainfall": 0
}

MOCK_SOIL_DATA = {
    "soil_type": "Loamy",
    "moisture": 45
}

@app.route('/')
def home():
    return render_template('simple_irrigation.html')

@app.route('/real-time-data')
def get_real_time_data():
    try:
        # Get latitude and longitude from request parameters
        latitude = request.args.get('latitude')
        longitude = request.args.get('longitude')
        
        # If coordinates are provided, fetch real weather data
        if latitude and longitude:
            logger.info(f"Fetching weather data for coordinates: {latitude}, {longitude}")
            
            # Call WeatherAPI.com API
            params = {
                'key': WEATHER_API_KEY,
                'q': f"{latitude},{longitude}",
                'aqi': 'no'
            }
            
            try:
                logger.info(f"Making request to WeatherAPI.com with params: {params}")
                response = requests.get(WEATHER_API_BASE_URL, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                weather_data = response.json()
                logger.info(f"Weather API response: {weather_data}")
                
                # Extract relevant data from WeatherAPI.com response
                temperature = weather_data['current']['temp_c']
                humidity = weather_data['current']['humidity']
                wind_speed = weather_data['current']['wind_kph'] / 3.6  # Convert km/h to m/s
                
                # Get rainfall (if available)
                rainfall = 0
                if 'precip_mm' in weather_data['current']:
                    rainfall = weather_data['current']['precip_mm']
                
                # Get soil type based on location (simplified)
                # In a real application, you would use a soil database API
                soil_type = determine_soil_type(float(latitude), float(longitude))
                
                logger.info(f"Successfully fetched weather data: {temperature}°C, {humidity}%, {wind_speed} m/s, {rainfall} mm")
                
                return jsonify({
                    "temperature": round(temperature, 1),
                    "humidity": humidity,
                    "wind_speed": round(wind_speed, 1),
                    "rainfall": round(rainfall, 1),
                    "soil_type": soil_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "WeatherAPI.com",
                    "location": f"{latitude}, {longitude}"
                })
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching weather data: {str(e)}")
                # Fall back to mock data
                return jsonify({
                    "temperature": MOCK_WEATHER_DATA["temperature"],
                    "humidity": MOCK_WEATHER_DATA["humidity"],
                    "wind_speed": MOCK_WEATHER_DATA["wind_speed"],
                    "rainfall": MOCK_WEATHER_DATA["rainfall"],
                    "soil_type": MOCK_SOIL_DATA["soil_type"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": f"Using fallback data due to API error: {str(e)}",
                    "source": "Mock data"
                })
        else:
            # No coordinates provided, use mock data
            logger.info("No coordinates provided, using mock data")
            return jsonify({
                "temperature": MOCK_WEATHER_DATA["temperature"],
                "humidity": MOCK_WEATHER_DATA["humidity"],
                "wind_speed": MOCK_WEATHER_DATA["wind_speed"],
                "rainfall": MOCK_WEATHER_DATA["rainfall"],
                "soil_type": MOCK_SOIL_DATA["soil_type"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Using default data (location not provided)",
                "source": "Mock data"
            })
            
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        return jsonify({"error": f"Failed to fetch real-time data: {str(e)}"}), 500

def determine_soil_type(latitude, longitude):
    """
    Determine soil type based on latitude and longitude.
    This is a simplified function - in a real application, you would use a soil database API.
    """
    # Simple rule-based approach based on latitude
    if latitude > 60:  # Northern regions
        return "Sandy"
    elif latitude > 30:  # Temperate regions
        return "Loamy"
    elif latitude > 0:  # Tropical regions
        return "Clay"
    else:  # Southern regions
        return "Red"

@app.route('/predict', methods=['POST'])
def predict_irrigation():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Log the received data for debugging
        logger.info(f"Received form data: {data}")
        
        # Get location data if available
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        # If location is provided, fetch real-time weather data
        if latitude and longitude:
            try:
                params = {
                    'key': WEATHER_API_KEY,
                    'q': f"{latitude},{longitude}",
                    'aqi': 'no'
                }
                
                logger.info(f"Making request to WeatherAPI.com for prediction with params: {params}")
                response = requests.get(WEATHER_API_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                
                weather_data = response.json()
                logger.info(f"Weather API response for prediction: {weather_data}")
                
                temperature = weather_data['current']['temp_c']
                
                # Update the temperature in the form data
                data['temperature'] = str(round(temperature, 1))
                logger.info(f"Updated temperature from weather API: {temperature}°C")
            except Exception as e:
                logger.error(f"Error fetching weather data for prediction: {str(e)}")
                # Continue with the user-provided temperature
        
        # Prepare prompt for Gemini
        prompt = f"""
        Based on the following parameters, provide irrigation recommendations:
        
        Soil Type: {data.get('soil_type')}
        Soil Moisture: {data.get('moisture')}%
        Crop Type: {data.get('crop_type')}
        Growth Stage: {data.get('growth_stage')}
        Temperature: {data.get('temperature')}°C
        Expected Rainfall: {data.get('rainfall')}mm
        Location: {latitude}, {longitude if latitude and longitude else 'Not provided'}
        
        Please provide:
        1. Amount of irrigation needed (in mm)
        2. Frequency of irrigation
        3. Best time for irrigation
        4. Recommended irrigation method
        5. Brief explanation of the recommendation
        """
        
        try:
            # Get response from Gemini
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Parse the response
            response_text = response.text
            
            # Extract structured data from the response
            # This is a simplified approach - in a real app, you'd want more robust parsing
            amount = "25-30"  # Default value
            frequency = "Every 3-4 days"  # Default value
            best_time = "Early morning"  # Default value
            method = "Drip irrigation"  # Default value
            
            # Try to extract values from the response text
            if "mm" in response_text:
                # Simple extraction - in a real app, use regex or better parsing
                amount = response_text.split("mm")[0].split()[-1]
            
            if "frequency" in response_text.lower():
                # Simple extraction
                freq_parts = response_text.lower().split("frequency")
                if len(freq_parts) > 1:
                    frequency = freq_parts[1].split(".")[0].strip()
            
            if "time" in response_text.lower():
                # Simple extraction
                time_parts = response_text.lower().split("time")
                if len(time_parts) > 1:
                    best_time = time_parts[1].split(".")[0].strip()
            
            if "method" in response_text.lower():
                # Simple extraction
                method_parts = response_text.lower().split("method")
                if len(method_parts) > 1:
                    method = method_parts[1].split(".")[0].strip()
            
            return jsonify({
                "amount": amount,
                "frequency": frequency,
                "best_time": best_time,
                "method": method,
                "explanation": response_text,
                "temperature": data.get('temperature'),
                "source": "WeatherAPI.com" if latitude and longitude else "User input",
                "location": f"{latitude}, {longitude}" if latitude and longitude else "Not provided"
            })
            
        except Exception as api_error:
            # If Gemini API fails, use fallback logic
            logger.error(f"Gemini API error: {str(api_error)}")
            
            # Fallback logic based on simple rules
            soil_type = data.get('soil_type', 'loamy')
            moisture = float(data.get('moisture', 50))
            crop_type = data.get('crop_type', 'wheat')
            growth_stage = data.get('growth_stage', 'vegetative')
            temperature = float(data.get('temperature', 25))
            rainfall = float(data.get('rainfall', 0))
            
            # Simple rule-based logic
            if moisture < 30:
                amount = "30-40"
                frequency = "Every 2-3 days"
            elif moisture < 50:
                amount = "20-30"
                frequency = "Every 3-4 days"
            else:
                amount = "10-20"
                frequency = "Every 4-5 days"
                
            if temperature > 30:
                best_time = "Early morning or late evening"
            else:
                best_time = "Morning"
                
            if soil_type == "sandy":
                method = "Drip irrigation"
            elif soil_type == "clay":
                method = "Sprinkler irrigation"
            else:
                method = "Surface irrigation"
                
            explanation = f"Based on {soil_type} soil with {moisture}% moisture, {crop_type} in {growth_stage} stage, with {temperature}°C temperature and {rainfall}mm expected rainfall."
            
            return jsonify({
                "amount": amount,
                "frequency": frequency,
                "best_time": best_time,
                "method": method,
                "explanation": explanation,
                "temperature": data.get('temperature'),
                "source": "WeatherAPI.com" if latitude and longitude else "User input",
                "location": f"{latitude}, {longitude}" if latitude and longitude else "Not provided"
            })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Failed to generate prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 