# # Without Voice Recog..
# import cv2
# import mediapipe as mp
# import pandas as pd
# import os
# import numpy as np
# import pickle
# import tkinter as tk
# from tkinter import messagebox

# def image_processed(hand_img):
#     # Image processing
#     # 1. Convert BGR to RGB
#     img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

#     # 2. Flip the img in Y-axis
#     img_flip = cv2.flip(img_rgb, 1)

#     # accessing MediaPipe solutions
#     mp_hands = mp.solutions.hands

#     # Initialize Hands
#     hands = mp_hands.Hands(static_image_mode=True,
#                            max_num_hands=1, min_detection_confidence=0.7)

#     # Results
#     output = hands.process(img_flip)

#     hands.close()

#     try:
#         data = output.multi_hand_landmarks[0]
#         # print(data)
#         data = str(data)

#         data = data.strip().split('\n')

#         garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

#         without_garbage = []

#         for i in data:
#             if i not in garbage:
#                 without_garbage.append(i)

#         clean = []

#         for i in without_garbage:
#             i = i.strip()
#             clean.append(i[2:])

#         for i in range(0, len(clean)):
#             clean[i] = float(clean[i])
#         return clean
#     except:
#         return np.zeros([1, 63], dtype=int)[0]

# # Load model
# with open('model.pkl', 'rb') as f:
#     svm = pickle.load(f)

# class App:
#     def __init__(self, window, window_title):
#         self.window = window
#         self.window.title(window_title)

#         # Camera Access Button
#         self.camera_button = tk.Button(window, text="Open Camera", command=self.open_camera)
#         self.camera_button.pack(pady=20)

#         # About Section
#         about_label = tk.Label(window, text="Indian Hand Sign Language Recognition :", font=("Helvetica", 18, "bold"))
#         about_label.pack()

#         # About Section
#         about_label = tk.Label(window, text="Indian Sign Language (ISL) is a visual language used by\nthe deaf and hard-of-hearing community in India.\nIt is a complete language, with its own \ngrammar and syntax, and is used to convey information \nthrough hand gestures, facial expressions, and body language.\n\n")
#         about_label.pack()

#         # About Section
#         about_label = tk.Label(window, text="About:")
#         about_label.pack()

#         about_text = tk.Text(window, height=5, width=50)
#         about_text.insert(tk.END, "Developed by:Syed Imtiyaz Ali")
#         about_text.pack()

#         # Camera Screen
#         self.canvas = tk.Canvas(window, width=800, height=350)
#         self.canvas.pack()

    
#     def open_camera(self):
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             messagebox.showerror("Error", "Cannot open camera")
#             return

#         while True:
#             ret, frame = cap.read()

#             if not ret:
#                 messagebox.showerror("Error", "Can't receive frame (stream end?). Exiting ...")
#                 break

#             data = image_processed(frame)
#             data = np.array(data)
#             y_pred = svm.predict(data.reshape(-1, 63))

#             font = cv2.FONT_HERSHEY_SIMPLEX
#             org = (50, 100)
#             fontScale = 3
#             color = (255, 0, 0)
#             thickness = 5

#             frame = cv2.putText(frame, str(y_pred[0]), org, font,
#                                 fontScale, color, thickness, cv2.LINE_AA)
#             cv2.imshow('frame', frame)

#             if cv2.waitKey(1) == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# # Create the main window
# root = tk.Tk()
# app = App(root, "Indian Hand Sign Language Recognition App")
# root.mainloop()


# # # With Voice  Recognition -
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import pickle
# # import cv2 as cv
# # import pyttsx3

# # def image_processed(hand_img):
# #     # Image processing
# #     img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
# #     img_flip = cv2.flip(img_rgb, 1)

# #     # accessing MediaPipe solutions
# #     mp_hands = mp.solutions.hands

# #     # Initialize Hands
# #     hands = mp_hands.Hands(static_image_mode=True,
# #                            max_num_hands=1, min_detection_confidence=0.7)

# #     # Results
# #     output = hands.process(img_flip)

# #     hands.close()

# #     try:
# #         data = output.multi_hand_landmarks[0]
# #         data = str(data)

# #         data = data.strip().split('\n')

# #         garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

# #         without_garbage = []

# #         for i in data:
# #             if i not in garbage:
# #                 without_garbage.append(i)

# #         clean = []

# #         for i in without_garbage:
# #             i = i.strip()
# #             clean.append(i[2:])

# #         for i in range(0, len(clean)):
# #             clean[i] = float(clean[i])
# #         return clean
# #     except:
# #         return np.zeros([1, 63], dtype=int)[0]

# # def text_to_speech(text):
# #     # Initialize the text-to-speech engine
# #     engine = pyttsx3.init()
# #     engine.say(text)
# #     engine.runAndWait()

# # # Load the SVM model
# # with open('model.pkl', 'rb') as f:
# #     svm = pickle.load(f)

# # cap = cv.VideoCapture(0)
# # if not cap.isOpened():
# #     print("Cannot open camera")
# #     exit()

# # prev_gesture = None  # Variable to store the previous detected gesture

# # while True:
# #     ret, frame = cap.read()

# #     if not ret:
# #         print("Can't receive frame (stream end?). Exiting ...")
# #         break

# #     data = image_processed(frame)
# #     data = np.array(data)
# #     y_pred = svm.predict(data.reshape(-1, 63))
# #     print(y_pred[0])

# #     if y_pred[0] != prev_gesture:
# #         # Display the predicted gesture on the frame
# #         font = cv2.FONT_HERSHEY_SIMPLEX
# #         org = (50, 100)
# #         fontScale = 3
# #         color = (255, 0, 0)
# #         thickness = 5
# #         frame = cv2.putText(frame, str(y_pred[0]), org, font, fontScale, color, thickness, cv2.LINE_AA)

# #         # Speak the predicted gesture
# #         text_to_speech(f'The detected gesture is {y_pred[0]}')

# #         # Update the previous gesture
# #         prev_gesture = y_pred[0]

# #     cv.imshow('frame', frame)
# #     if cv.waitKey(1) == ord('q'):
# #         break

# # cap.release()
# # cv.destroyAllWindows()


from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

def image_processed(hand_img):
    # Image processing code
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()
    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = [i for i in data if i not in garbage]
        clean = [i.strip()[2:] for i in without_garbage]
        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

def camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Cannot open camera"

    while True:
        ret, frame = cap.read()

        if not ret:
            return "Can't receive frame (stream end?)"

        data = image_processed(frame)
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1, 63))

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)
        fontScale = 3
        color = (255, 0, 0)
        thickness = 5

        frame = cv2.putText(frame, str(y_pred[0]), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
