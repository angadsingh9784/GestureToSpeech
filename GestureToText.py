import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import StringVar
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image, ImageTk
import os
import time

# Initialize the classifier with your model (update the path to your model file)
model_path = 'C:/Users/user/PycharmProjects/CapstoneProject/Model/keras_model.h5'
labels_path = 'C:/Users/user/PycharmProjects/CapstoneProject/Model/labels.txt'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

classifier = Classifier(model_path, labels_path)
confidence_threshold = 0.85  # Set a confidence threshold to ensure stable gestures are detected
labels = ["Can you write it down?",
          "I am deaf/hard of hearing",
          "I use sign language to communicate",
          "Please face me while talking",
          "Is there an interpreter available?"]  # Example word labels  # Example labels

# OpenCV setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# GUI setup
root = tk.Tk()
root.attributes('-fullscreen', True)
root.title("Gesture to Text Converter")

# Display welcome message for 5 seconds
welcome_label = tk.Label(root, text="Welcome to the Sign Speak Project", font=("Helvetica", 24))
welcome_label.pack()
root.update()
time.sleep(5)
welcome_label.pack_forget()

# Tkinter variables to hold data
sentence = StringVar()
sentence.set("")
previous_letter = None
last_gesture_time = time.time()
letter_pause_time = time.time()

# GUI elements
frame = tk.Frame(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
frame.pack(expand=True)
frame.place(relx=0.5, rely=0.5, anchor='center')

video_label = tk.Label(frame)
video_label.grid(row=0, column=0, columnspan=4)

sentence_label = tk.Label(frame, textvariable=sentence, font=("Helvetica", 24), anchor='center')
sentence_label.grid(row=1, column=0, columnspan=4)

clear_button = tk.Button(frame, text="Clear", command=lambda: sentence.set(""), font=("Helvetica", 16))
clear_button.grid(row=2, column=1, pady=10)

exit_button = tk.Button(frame, text="Exit", command=root.quit, font=("Helvetica", 16))
exit_button.grid(row=2, column=2, pady=10)


def update_sentence(word):
    """Update the sentence with the detected letter."""
    current_sentence = sentence.get()
    if current_sentence:
        sentence.set(current_sentence + " " + word)
    else:
        sentence.set(word)


# Function to update the video feed
def update_video():
    global previous_letter, last_gesture_time, letter_pause_time
    success, img = cap.read()
    if success:
        img_output = img.copy()
        hands, img = detector.findHands(img)
        current_time = time.time()

        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                img_crop = img[y1:y2, x1:x2]

                aspect_ratio = h / w

                if aspect_ratio > 1:
                    k = imgSize / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                else:
                    k = imgSize / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)

                confidence = prediction[index]
                word = labels[index]

                if confidence > confidence_threshold and (
                        word != previous_letter or (current_time - last_gesture_time > 5)) and (
                        current_time - letter_pause_time > 5):
                    update_sentence(word)
                    previous_letter = word
                    last_gesture_time = current_time
                    letter_pause_time = current_time

                # Displaying the label on the video feed
                cv2.rectangle(img_output, (x - offset, y - offset - 90),
                              (x - offset + 80, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(img_output, word, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(img_output, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            except Exception as e:
                print(f"Error during processing: {e}")

        # Convert the image to PhotoImage for Tkinter
        img_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_video)


# Start video feed update
update_video()

# Start the GUI loop
root.mainloop()

# Release the video capture when the program closes
cap.release()
cv2.destroyAllWindows()
