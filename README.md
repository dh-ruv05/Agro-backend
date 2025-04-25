# Agro Nexus Backend

This is the backend service for the Agro Nexus application, providing APIs for plant disease detection, crop recommendations, and agricultural data analysis.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
MONGODB_URI=your_mongodb_uri
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
```

4. Run the application:
```bash
python app.py
```

The server will start on port 5000 by default.

## API Endpoints

- `/test` - Health check endpoint
- Additional endpoints documentation coming soon

## Deployment

This backend is configured for deployment on Railway or Render. Make sure to set up the environment variables in your deployment platform. 