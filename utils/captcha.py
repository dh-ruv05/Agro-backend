import requests
from flask import current_app

def verify_captcha(captcha_response):
    """
    Verify the reCAPTCHA response with Google's API
    """
    if not captcha_response:
        return False
        
    payload = {
        'secret': current_app.config['RECAPTCHA_SECRET_KEY'],
        'response': captcha_response
    }
    
    try:
        response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data=payload,
            timeout=5
        )
        result = response.json()
        return result.get('success', False)
    except Exception as e:
        current_app.logger.error(f"CAPTCHA verification failed: {str(e)}")
        return False 