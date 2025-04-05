# Face-recognition-attendance-system
It is an advanced system designed to automate and streamline attendance management using facial recognition technology

# Project Overview
The system:
- Captures a student's face through a webcam using OpenCV.
- Matches the captured face with pre-encoded images stored in Firebase.
- Updates the student's attendance in real-time based on face recognition.
- Uploads student images to Firebase Storage.
- Displays student information on a GUI, including their attendance count, name, major, and other details.

# Objective
- ✅ Develop a face recognition based attendance system for automated and contactless attendance tracking.
- ✅ Utilize OpenCV and deep learning for accurate facial detection and recognition.
- ✅ Enhance security and efficiency by eliminating manual errors and preventing proxy attendance.

# Project Differentiation:
- Hybrid Storage Model (Cloud + Local Database): Unlike systems that rely solely on cloud storage, we implement a hybrid model where encrypted facial data is stored locally for speed while backup records are maintained securely on the cloud for redundancy.
- Embeddings for recognition: Instead of directly comparing the raw images(pixels), we use embeddings which are mathematical representation od data in form of vectors, which speeds up face recognition for larger dataset.
- Leveraging DLIB: Dlib leverages a deep learning model that has already been trained on a large dataset of faces. This model has learned to extract meaningful features from a face image and represent them in a way that enables comparison and recognition.

# Technologies Used
- Python: Main programming language.
- OpenCV: For image processing and face detection.
- face_recognition: For facial feature recognition and comparison.
- Firebase: Firebase Realtime Database for storing student data.
- dotenv: For loading sensitive environment variables such as Firebase credentials and database URL.
- Pickle: For saving and loading face encodings for fast matching.

  




