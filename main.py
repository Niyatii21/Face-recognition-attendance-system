import os
import pickle
import numpy as np
import cv2
import cvzone
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import config
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate(config.FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {
    'databaseURL': config.FIREBASE_DATABASE_URL
})

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackground = cv2.imread('Resources/background.png')

# Load mode images
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load encoding file
print("LOADING ENCODE FILE...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownIds
print("ENCODE FILE LOADED")

# Load student images from the "Images" folder
folderPath = "Images"
studentImages = {}
for img_file in os.listdir(folderPath):
    img_id = os.path.splitext(img_file)[0]
    img_path = os.path.join(folderPath, img_file)
    studentImages[img_id] = cv2.imread(img_path)
    if studentImages[img_id] is None:
        print(f"Warning: Failed to load image {img_path}")

modeType = 0
counter = 0
id = -1
studentInfo = None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Resize and convert to RGB for face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find faces
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    # Place webcam feed on background
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurrentFrame:  # If faces are detected
        for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox=bbox, rt=0)
                id = studentIds[matchIndex]

                # Show loading message when a face is recognized
                if counter == 0:
                    cvzone.putTextRect(imgBackground, 'Loading...', (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)  # Brief delay to show the loading message
                    counter = 1
                    modeType = 1

    if counter != 0:
        if counter == 1:
            # Fetch student data from Firebase
            studentInfo = db.reference(f'Students/{id}').get()
            print("Type of studentInfo:", type(studentInfo))
            print("studentInfo:", studentInfo)
            if not isinstance(studentInfo, dict) or 'total_Attendance' not in studentInfo:
                print(f"Error: Invalid studentInfo for ID {id}. Type: {type(studentInfo)}")
                studentInfo = None
                modeType = 0
                counter = 0
                continue

            # Check time elapsed since last attendance
            try:
                attendanceTime = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - attendanceTime).total_seconds()

                if secondsElapsed > 30:  # If more than 30 seconds have passed
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_Attendance'] += 1
                    ref.child('total_Attendance').set(studentInfo['total_Attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3  # Mode to indicate recent attendance
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
            except Exception as e:
                print(f"Error processing attendance time: {e}")
                studentInfo = None
                modeType = 0
                counter = 0
                continue

        # Switch to mode 2 (e.g., "Attendance Marked") between frames 10 and 20
        if 10 < counter <= 20:
            modeType = 2

        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Display student details if not in "recent attendance" mode
        if modeType != 3:
            if counter <= 10:
                try:
                    # Display student details
                    cv2.putText(imgBackground, str(studentInfo['total_Attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['name']), (808, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    if id in studentImages:
                        studentImage = studentImages[id]
                        studentImageResized = cv2.resize(studentImage, (216, 216), interpolation=cv2.INTER_AREA)
                        imgBackground[175:175 + 216, 909:909 + 216] = studentImageResized
                    else:
                        print(f"No image found for ID {id} in studentImages")

                except Exception as e:
                    print(f"Error drawing text or image: {e}")
                    studentInfo = None
                    modeType = 0
                    counter = 0

        counter += 1

        # Reset after 20 frames
        if counter > 20:
            counter = 0
            studentInfo = None
            modeType = 0
            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        # Reset to initial state if no face is detected
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()