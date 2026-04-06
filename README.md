# Python_project

# 📷 Face Recognition Attendance Monitoring System

A real-time, AI-powered attendance monitoring system built with Python, OpenCV, and Tkinter. It automatically detects and recognizes student faces via webcam and marks attendance — no manual entry needed.

---

## ✨ Features

- 🔐 **Admin Login** — Secure access with username and password
- 👤 **Student Registration** — Add students with ID, name, age, and gender
- 📷 **Face Photo Capture** — Captures 100 grayscale face samples per student
- 🧠 **Model Training** — Trains an LBPH face recognizer and saves it to disk
- ✅ **Real-time Attendance** — Recognizes faces via webcam and marks attendance automatically
- 🚫 **Duplicate Prevention** — Each student can only be marked once per day
- 📊 **Attendance Report** — View and filter attendance records by date
- 🎨 **Modern Dark UI** — Clean, styled Tkinter interface

---

## 🖥️ Screenshots

> Dashboard | Student Management | Attendance Report

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| OpenCV (`cv2`) | Face detection & recognition |
| Tkinter | GUI interface |
| Pandas | Excel data handling |
| NumPy | Numerical operations |
| OpenPyXL | Read/write `.xlsx` files |

---

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

---

## 🚀 How to Run

```bash
python Attendance_monitoring.py
```

**Default login credentials:**
- Username: `admin`
- Password: `password`

---

## 📋 How to Use

1. **Register a Student** → Go to Student Management → Enter ID, Name, Age, Gender
2. **Capture Photos** → Go to Capture Face Photos → Enter Student ID → Allow camera to capture 100 samples
3. **Train the Model** → Click "Train Model" on the Dashboard
4. **Take Attendance** → Click "Take Attendance" → Camera will recognize faces and mark attendance automatically
5. **View Reports** → Click "Attendance Report" → Filter by date if needed

---

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

---

## 👩‍💻 Author

**Meenakshi** — [@meenakship0805](https://github.com/meenakship0805)

---
