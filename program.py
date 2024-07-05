import cv2 as cv
import numpy as np
import os
import csv
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load(r"faces_embeddings_done_4classes_new.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier(r"haarcascade_frontalface_default (1).xml")
model = pickle.load(open(r"svm_model_160x160_new.pkl", 'rb'))
# Create a named window with full screen flag
cv.namedWindow('Face Recognition:', cv.WINDOW_FULLSCREEN)

# Open the video capture device (webcam)
cap = cv.VideoCapture(0)

# Set to track unique faces
unique_faces = set()

# CSV file setup
csv_file_path = r"detected_faces.csv"
csv_unique_faces_path = r"unique_faces.csv"

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time of Entry"])

with open(csv_unique_faces_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name"])

# Function to log unique face to CSV
def log_face(name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time])

def log_unique_face(name):
    with open(csv_unique_faces_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name])


# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        break
    
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    frame_faces = []
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160)) # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        print(max(model.predict_proba(ypred)[0]))
        if(max(model.predict_proba(ypred)[0]) > 0.30):
            final_name = encoder.inverse_transform(face_name)[0]
        else:
            final_name = 'Unknown'

        if final_name!='Unknown' and final_name not in unique_faces:
            unique_faces.add(final_name)  # Track unique face names
            log_unique_face(final_name) 

        if final_name!='Unknown' and final_name not in frame_faces:
            frame_faces.append(final_name)
            log_face(final_name)  # Log to CSV

        if(final_name == 'Unknown'):
            # unique_faces.add(final_name)  # Track unique face names
            # log_unique_face(final_name) 
            frame_faces.append(final_name)
            log_face(final_name)  # Log to CSV

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Display the count of unique faces in frame detected
    cv.putText(frame, str(len(set(frame_faces))) + ' Faces In Frame', (200, 440), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 0), 2, cv.LINE_AA)

    # Display the count of unique faces detected
    cv.putText(frame, str(len(unique_faces)) + ' Unique Faces Detected', (160, 475), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 2, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv.destroyAllWindows()
