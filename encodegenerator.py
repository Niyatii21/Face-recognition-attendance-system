# Purpose of the Code
# Input: A folder of images, where each image corresponds to a student.
# Output:
# A list of face encodings for the images.
# A list of student IDs (based on the filenames).
# These encodings can later be used for tasks like face recognition, attendance marking,
# or verifying if a given face matches a stored student image.
#encodegenerator.py
import cv2
import face_recognition
import pickle
import os

# importing the student images
folderPath = 'Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    # print(path)
    # print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])
# print(len(imgList))
print(studentIds)

#loop through all the images and encode every single image
def findEncodingd(imagesLists):
    #change the colourspace bgr to rgb --> opencv uses bgr --->facerecog uses rgb
    encodeList = []
    for img in imagesLists:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode  = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("ENCODING STARTED...")
encodeListKnown = findEncodingd(imgList)
# print(encodeListKnown)
encodeListKnownIds =  [encodeListKnown , studentIds]
print("ENCODING COMPLETE")

#storing the encoding along with the student id in a pickle file
#'wb': Write mode in binary format (required for saving data using pickle).
file = open ("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownIds,file)
file.close()
print("FILE SAVED")
