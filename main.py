import pyttsx3
import speech_recognition as sr
import subprocess

engine = pyttsx3.init()

def execute_file(file_path):
    subprocess.run(["python", file_path])

recognizer = sr.Recognizer()

def listen_for_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("Command:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        engine.say("Sorry, I didn't understand that.")
        engine.runAndWait()
        return None
    except sr.RequestError:
        print("Sorry, I couldn't request results. Please check your internet connection.")
        engine.say("Sorry, I couldn't request results. Please check your internet connection.")
        engine.runAndWait()
        return None

while True:
    command = listen_for_command()
    if command == "stop":
        print("Stopping execution.")
        engine.say("Stopping execution.")
        engine.runAndWait()
        break
    elif command == "capture":
        engine.say("Executing capture.")
        engine.runAndWait()
        execute_file("D:/IFP/face recognition/capture.py")
    elif command == "detect face":
        engine.say("Executing face detection.")
        engine.runAndWait()
        execute_file("D:/IFP/face recognition/capture.py")
    elif command == "detect object":
        engine.say("Executing object detection.")
        engine.runAndWait()
        execute_file("D:/IFP/Object_Detect/object_detect.py")
    elif command == "read text":
        engine.say("Reading text.")
        engine.runAndWait()
        execute_file("D:/IFP/sampocr.py")
