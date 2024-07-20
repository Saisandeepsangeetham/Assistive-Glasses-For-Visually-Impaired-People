import os
import face_recognition
import cv2
import numpy as np
import pyttsx3
import requests

def initialize_text_to_speech():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

def capture_img(stream_url,engine):
    try:
        response = requests.get(stream_url, timeout=10)
        if response.status_code == 200:
            image_array = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, -1)
            return image
        else:
            print("Failed to capture image. Status code:", response.status_code)
            engine.say("Failed to capture image")
            engine.runAndWait()
            return None
    except requests.RequestException as e:
        print("An error occurred while capturing the image:", e)
        engine.say("Error occurred while capturing the image")
        engine.runAndWait()
        return None

def load_known_faces(known_faces_path):
    known_faces_encodings = {}
    for filename in os.listdir(known_faces_path):
        image_path = os.path.join(known_faces_path, filename)
        image = face_recognition.load_image_file(image_path)
        
        face_encoding = face_recognition.face_encodings(image)
        
        if len(face_encoding) > 0:
            known_faces_encodings[filename] = face_encoding[0]
    return known_faces_encodings

def recognize_faces(known_faces_encodings, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    recognized_names = []
    recognized_name = 'Unknown'

    for face_encoding in face_encodings:
        for name, known_encodings in known_faces_encodings.items():
            matches = face_recognition.compare_faces([np.array(known_encodings)], np.array(face_encoding))
            match_score = sum(matches) / len(matches)
            
            if match_score > 0.5: 
                recognized_name = name.split(".")[0]
                break  
        recognized_names.append(recognized_name)
    if len(recognized_names)==0:
        recognized_names.append("Not known")
    return recognized_names

def main():
    engine = initialize_text_to_speech()
#change the path accordingly.
    known_faces_path = 'D:/IFP/face recognition/Datasets/'
    known_faces_encodings = load_known_faces(known_faces_path)

    stream_url = "http://192.168.219.9/capture"
    frame = capture_img(stream_url,engine)

    if frame is not None:
        recognized_names = recognize_faces(known_faces_encodings, frame)
        speech_text = " ".join(recognized_names)
        
        engine.say("Detected faces")
        engine.runAndWait()
        engine.say(speech_text)
        engine.runAndWait()
        
        print("Recognized names:", recognized_names)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
