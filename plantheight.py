import numpy as np
import cv2
import io
from PIL import Image
import base64

BASE_HEIGHT = 38.5

def calculate_plant_height(file_storage):
    # Read the image from uploaded file (werkzeug FileStorage)
    in_memory_file = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    image_array = np.array(img)
    blurred_frame = cv2.blur(image_array, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Define color threshold (tuned for green)
    low_green = np.array([30, 10, 50])
    high_green = np.array([135, 255, 200])

    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    # Morphological adjustments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"error": "No plant detected"}

    biggest = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(image_array, [biggest], -1, (0, 0, 0), 1)

    # Create mask and extract plant
    blank_mask = np.zeros(image_array.shape, dtype=np.uint8)
    cv2.fillPoly(blank_mask, [biggest], (255, 255, 255))
    blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)

    result = cv2.bitwise_and(image_array, image_array, mask=blank_mask)
    positions = np.nonzero(result)

    if positions[0].size == 0:
        return {"error": "No green area found"}

    top = positions[0].min()
    bottom = positions[0].max()

    ratio = (bottom - top) / img.shape[0]
    height = ratio * BASE_HEIGHT

    # Convert image to base64 for HTML display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "height": round(height, 2),
        "image_base64": encoded_img
    }
