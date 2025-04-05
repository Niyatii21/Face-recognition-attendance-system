import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
# from dotenv import load_dotenv
#addtodatabse.py

# Load environment variables from .env file
# load_dotenv()

# Initialize Firebase Admin SDK with credentials and database URL
cred = credentials.Certificate("Serviceaccountkey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-recog-abe92-default-rtdb.firebaseio.com/"
})

# Reference to the "Students" node in Firebase Realtime Database
ref=db.reference('Students')

# Student data to be uploaded to Firebase
data = {
"21109":
        {
            "name" : "Niyati Negi",
            "major": "Btech",
            "starting year" :2021,
            "total_Attendance": 6,
            "standing":"A",
            "year":4,
            "last_attendance_time":"2025-01-24 00:54:34"
        },
"24105":
        {
            "name" : "Mudit Jain ",
            "major": "Btech",
            "starting year" :2021,
            "total_Attendance": 7,
            "standing":"A",
            "year":4,
            "last_attendance_time":"2025-01-25 00:54:34"
        },

"852741":
        {
            "name" : "Emly Blunt",
            "major": "BArch",
            "starting year" :2021,
            "total_Attendance": 6,
            "standing":"B",
            "year":4,
            "last_attendance_time":"2025-01-24 00:54:34"
        },
"963852":
        {
            "name" : "Elon Musk",
            "major": "BArch",
            "starting year" :2023,
            "total_Attendance": 6,
            "standing":"C ",
            "year":4,
            "last_attendance_time":"2025-01-24 00:54:34"
        },



"7321100":
        {
            "name" : "Keshav Chauhan",
            "major": "BTech",
            "starting year" :2023,
            "total_Attendance": 6,
            "standing":"C ",
            "year":4,
            "last_attendance_time":"2025-01-24 00:54:34"
        },


}

# Uploading student data to Firebase
for key, value in data.items():
    try:
        ref.child(key).set(value)
        print(f"Successfully uploaded data for student ID: {key}")
    except Exception as e:
        print(f"Error uploading data for student ID: {key}. Error: {e}")