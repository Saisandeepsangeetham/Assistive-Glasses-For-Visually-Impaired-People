import cv2
import pyttsx3
import numpy as np
import requests
import time

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

def load_yolo_model(weights_path, cfg_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def load_classes(labels_path):
    with open(labels_path, "r") as f:
        classes = f.read().strip().split("\n")
    return classes

def detect_objects(net, image, classes):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detected_objects.append(classes[class_ids[i]])

    return detected_objects, image

def main():
    start = time.time()
    weights_path = "/home/ssn/IFP/Object_Detect/Model/yolov4.weights"
    cfg_path = "/home/ssn/IFP/Object_Detect/Model/yolov4.cfg"
    labels_path = "/home/ssn/IFP/Object_Detect/Model/coco.names"
    
    engine = initialize_text_to_speech()
    net = load_yolo_model(weights_path, cfg_path)
    classes = load_classes(labels_path)

    # stream_url = 'http://192.168.50.9/capture'
    # image = capture_img(stream_url)
    image = cv2.imread('/home/ssn/IFP/Object_Detect/20240714133114.jpg')

    if image is not None:
        detected_objects, annotated_image = detect_objects(net, image, classes)
        
        if detected_objects:
            detected_objects_text = ", ".join(detected_objects)
            print(f"Detected objects: {detected_objects_text}")
            speech_text = f"I could see the objects: {detected_objects_text}"
            
            engine.say(speech_text)
            engine.runAndWait()

        end = time.time()
        print("Execution time:", end - start)
        
        cv2.imshow('Detected Objects', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture image.")
        engine.say("Failed to capture image")
        engine.runAndWait()

if __name__ == "__main__":
    main()
