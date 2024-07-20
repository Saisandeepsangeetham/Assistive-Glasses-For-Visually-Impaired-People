import numpy as np
import cv2
import requests
import os

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

def save_image(image, save_path):
    if image is not None:
        cv2.imwrite(save_path, image)
        print("Image saved successfully at:", save_path)
    else:
        print("Failed to save image.")

def main():
    stream_url = "http://192.168.219.9/capture"
    name = input("Enter the name:")
    name = name+".jpg"
    save_path = os.path.join("/home/ssn/IFP/Face_detection/Datasets/",name)

    image = capture_img(stream_url)
    save_image(image, save_path)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
