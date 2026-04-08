# Python_project

# 📷 Face Recognition Attendance Monitoring System

A real-time, AI-powered attendance monitoring system built with Python, OpenCV, and Tkinter. It automatically detects and recognizes student faces via webcam and marks attendance — no manual entry needed.



## ✨ Features

- 🔐 **Admin Login** — Secure access with username and password
- 👤 **Student Registration** — Add students with ID, name, age, and gender
- 📷 **Face Photo Capture** — Captures 100 grayscale face samples per student
- 🧠 **Model Training** — Trains an LBPH face recognizer and saves it to disk
- ✅ **Real-time Attendance** — Recognizes faces via webcam and marks attendance automatically
- 🚫 **Duplicate Prevention** — Each student can only be marked once per day
- 📊 **Attendance Report** — View and filter attendance records by date
- 🎨 **Modern Dark UI** — Clean, styled Tkinter interface



## 🖥️ Screenshots

> Dashboard | Student Management | Attendance Report
## 🖥️ Screenshots

### 🔐 Login Screen
<img width="398" height="350" alt="Login" src="https://github.com/user-attachments/assets/ef9fcf4e-7d52-4d31-87fd-4323b9fb40ef" />

### 🏠 Dashboard
<img width="431" height="433" alt="Dashboard" src="https://github.com/user-attachments/assets/0324c523-d212-40ab-9358-cd96c7ef1419" />

### 👤 Student Management
<img width="524" height="592" alt="Student Management" src="https://github.com/user-attachments/assets/78062f09-3254-4975-9f35-e210fa08e225" />

### 📷 Capture Face Photos
<img width="406" height="292" alt="Capture Photos" src="https://github.com/user-attachments/assets/8fb189b2-7d35-4dd3-99ec-a7dbcd02c315" />

### ✅ Take Attendance
<img width="480" height="383" alt="Take Attendance" src="https://github.com/user-attachments/assets/6a4b088f-40d5-459c-8268-5835dbdae280" />

### 📊 Attendance Report
<img width="600" height="532" alt="Attendance Report" src="https://github.com/user-attachments/assets/b237e142-3dbb-466c-ac77-eb44fc73836e" />






## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| OpenCV (`cv2`) | Face detection & recognition |
| Tkinter | GUI interface |
| Pandas | Excel data handling |
| NumPy | Numerical operations |
| OpenPyXL | Read/write `.xlsx` files |



## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/meenakship0805/Python_project.git
cd Python_project
```

### 2. Install dependencies
```bash
pip install opencv-contrib-python pandas openpyxl numpy
```

> ⚠️ Make sure to install `opencv-contrib-python` (NOT `opencv-python`) — it includes the face recognition module.



## 🚀 How to Run

```bash
python Attendance_monitoring.py
```

**Default login credentials:**
- Username: `admin`
- Password: `password`



## 📋 How to Use

1. **Register a Student** → Go to Student Management → Enter ID, Name, Age, Gender
2. **Capture Photos** → Go to Capture Face Photos → Enter Student ID → Allow camera to capture 100 samples
3. **Train the Model** → Click "Train Model" on the Dashboard
4. **Take Attendance** → Click "Take Attendance" → Camera will recognize faces and mark attendance automatically
5. **View Reports** → Click "Attendance Report" → Filter by date if needed



## 📁 Project Structure

```
Python_project/
│
├── Attendance_monitoring.py   # Main application
├── student_data.xlsx          # Registered students database
├── attendance_records.xlsx    # Attendance records
├── trained_model.yml          # Saved face recognition model (auto-generated)
├── student_photos/            # Captured face images (auto-generated)
└── README.md
```



## 👩‍💻 Author

**Meenakshi** — [@meenakship0805](https://github.com/meenakship0805)
