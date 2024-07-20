import requests
import numpy as np
import cv2
import easyocr
import pyttsx3
import time
from multiprocessing import Pool

def initialize_text_to_speech():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

def capture_img(stream_url):
    try:
        response = requests.get(stream_url, timeout=10)
        if response.status_code == 200:
            image_array = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, -1)
            return image
        else:
            print("Failed to capture image. Status code:", response.status_code)
            return None
    except requests.RequestException as e:
        print("An error occurred while capturing the image:", e)
        return None

def perform_ocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    return result

def main():
    stream_url = 'http://192.168.50.9/capture'
    engine = initialize_text_to_speech()

    start = time.time()
    image = capture_img(stream_url)
    end = time.time()
    print(f"Image capture time: {end - start} seconds")

    if image is not None:
        start = time.time()
        with Pool(processes=4) as pool:
            results = pool.map(perform_ocr, [image])
        end = time.time()
        print(f"OCR processing time: {end - start} seconds")

        detected_text = []
        for result in results:
            for detection in result:
                detected_text.append(detection[1])
        
        detected_text_str = " ".join(detected_text)
        print("Detected text:", detected_text_str)
        
        if detected_text:
            engine.say("Detected text")
            engine.runAndWait()
            engine.say(detected_text_str)
            engine.runAndWait()
    else:
        print("Failed to capture image.")
        engine.say("Failed to capture image")
        engine.runAndWait()

if __name__ == "__main__":
    main()
