#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import tkinter as tk
from tkinter import messagebox
import os
import pandas as pd
import numpy as np
from datetime import datetime


# In[8]:


import cv2
import tkinter as tk
from tkinter import messagebox
import os
import pandas as pd
import numpy as np
from datetime import datetime

def add_student_to_excel(student_number, student_full_name, student_age, student_gender):
    student_data = pd.DataFrame({
        'student_id': [int(student_number)],
        'student_name': [student_full_name],
        'age': [student_age],
        'gender': [student_gender]
    })
    if not os.path.exists('student_data.xlsx'):
        student_data.to_excel('student_data.xlsx', index=False)
    else:
        existing_data = pd.read_excel('student_data.xlsx')
        updated_data = pd.concat([existing_data, student_data], ignore_index=True)
        updated_data.to_excel('student_data.xlsx', index=False)

def mark_attendance_in_excel(student_number, student_full_name):
    attendance_data = pd.DataFrame({
        'student_id': [student_number],
        'student_name': [student_full_name],
        'timestamp': [pd.Timestamp.now()]
    })
    if not os.path.exists('attendance_records.xlsx'):
        attendance_data.to_excel('attendance_records.xlsx', index=False)
    else:
        existing_data = pd.read_excel('attendance_records.xlsx')
        updated_data = pd.concat([existing_data, attendance_data], ignore_index=True)
        updated_data.to_excel('attendance_records.xlsx', index=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_faces(image_frame):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def train_recognizer(image_samples, label_ids):
    recognizer.train(image_samples, np.array(label_ids))

def get_student_name(dataframe, student_number):
    student = dataframe[dataframe['student_id'] == student_number]
    if not student.empty:
        return student['student_name'].values[0]
    return "Unknown"

def recognize_face(image_frame, student_dataframe):
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY) if len(image_frame.shape) > 2 else image_frame
    id_, confidence = recognizer.predict(gray)
    threshold = 60
    if confidence < threshold:
        student_name = get_student_name(student_dataframe, id_)
        if student_name != "Unknown":
            mark_attendance_in_excel(id_, student_name)
        return student_name
    return "Unknown"

def load_training_data(directory_path):
    image_samples = []
    label_ids = []
    for person_name in os.listdir(directory_path):
        person_directory = os.path.join(directory_path, person_name)
        if os.path.isdir(person_directory):
            label = int(person_name)
            for filename in os.listdir(person_directory):
                filepath = os.path.join(person_directory, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    image_samples.append(img)
                    label_ids.append(label)
    return image_samples, label_ids

def login():
    entered_username = username_entry.get()
    entered_password = password_entry.get()
    if entered_username == "admin" and entered_password == "password":
        messagebox.showinfo("Login Successful", "Welcome, Admin!")
        home_page()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def home_page():
    root.destroy()
    home = tk.Tk()
    home.title("Home Page")
    home.geometry("400x300")

    tk.Button(home, text="Student Management", command=student_management).pack()
    tk.Button(home, text="Train Photo Samples", command=train_samples).pack()
    tk.Button(home, text="Take Attendance", command=take_attendance).pack()
    tk.Button(home, text="Attendance Report", command=attendance_report).pack()

    home.mainloop()

def student_management():
    student_window = tk.Toplevel()
    student_window.title("Student Management")
    student_window.geometry("400x200")

    tk.Label(student_window, text="Student Name:").grid(row=0, column=0)
    student_name_entry = tk.Entry(student_window)
    student_name_entry.grid(row=0, column=1)
    
    tk.Label(student_window, text="Student ID:").grid(row=1, column=0)
    student_id_entry = tk.Entry(student_window)
    student_id_entry.grid(row=1, column=1)
    
    tk.Button(student_window, text="Add Student", command=lambda: add_student_to_excel(student_id_entry.get(), student_name_entry.get(), '', '')).grid(row=2, column=0)

    student_window.mainloop()

def train_samples():
    def capture_photos():
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                student_dir = f"student_photos/{student_id_entry.get()}"
                os.makedirs(student_dir, exist_ok=True)
                cv2.imwrite(f"{student_dir}/{count}.jpg", roi_gray)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Capture Photos', frame)
            if cv2.waitKey(100) & 0xFF == ord('q') or count >= 100:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    train_window = tk.Toplevel()
    train_window.title("Train Photo Samples")
    train_window.geometry("400x200")
    
    tk.Label(train_window, text="Student ID:").pack()
    student_id_entry = tk.Entry(train_window)
    student_id_entry.pack()
    
    tk.Button(train_window, text="Capture Photos", command=capture_photos).pack()
    
    train_window.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Login")
    root.geometry("300x150")
    
    tk.Label(root, text="Username:").pack()
    username_entry = tk.Entry(root)
    username_entry.pack()
    
    tk.Label(root, text="Password:").pack()
    password_entry = tk.Entry(root, show="*")
    password_entry.pack()
    
    tk.Button(root, text="Login", command=login).pack()
    root.mainloop()


# In[ ]:





# In[ ]:




