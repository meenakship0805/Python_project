#!/usr/bin/env python
# coding: utf-8
"""
Complete Face Recognition Attendance Monitoring System
======================================================
Features:
- Admin login
- Student registration with full details
- Face photo capture (100 samples per student)
- Model training with persistence (save/load)
- Real-time face recognition attendance
- Duplicate attendance prevention (per session)
- Attendance report viewer
- Modern, styled Tkinter UI
"""

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
import os
import pandas as pd
import numpy as np
from datetime import datetime, date

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
STUDENT_FILE      = "student_data.xlsx"
ATTENDANCE_FILE   = "attendance_records.xlsx"
PHOTOS_DIR        = "student_photos"
MODEL_FILE        = "trained_model.yml"
ADMIN_USERNAME    = "admin"
ADMIN_PASSWORD    = "password"   # change in production
CONFIDENCE_THRESH = 60           # lower = stricter recognition
PHOTOS_PER_STUDENT = 100

# ─────────────────────────────────────────────
#  OPENCV SETUP
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False   # global flag

def _load_model_if_exists():
    """Load persisted model on startup."""
    global model_trained
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        model_trained = True

# ─────────────────────────────────────────────
#  EXCEL HELPERS
# ─────────────────────────────────────────────
def add_student_to_excel(student_id, name, age, gender):
    """Add a new student record. Prevents duplicate IDs."""
    row = pd.DataFrame({
        "student_id":   [int(student_id)],
        "student_name": [name],
        "age":          [int(age) if age else ""],
        "gender":       [gender]
    })
    if not os.path.exists(STUDENT_FILE):
        row.to_excel(STUDENT_FILE, index=False)
        return True, "Student added successfully."

    df = pd.read_excel(STUDENT_FILE)
    if int(student_id) in df["student_id"].values:
        return False, f"Student ID {student_id} already exists."

    df = pd.concat([df, row], ignore_index=True)
    df.to_excel(STUDENT_FILE, index=False)
    return True, "Student added successfully."


def get_all_students():
    """Return DataFrame of all students, or empty DataFrame."""
    if os.path.exists(STUDENT_FILE):
        return pd.read_excel(STUDENT_FILE)
    return pd.DataFrame(columns=["student_id", "student_name", "age", "gender"])


def get_student_name(student_id):
    df = get_all_students()
    row = df[df["student_id"] == student_id]
    return row["student_name"].values[0] if not row.empty else "Unknown"


def mark_attendance(student_id, student_name):
    """Mark attendance; returns False if already marked today."""
    now = datetime.now()
    today = date.today().isoformat()

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        # Check same student, same day
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        already = df[(df["student_id"] == student_id) & (df["date"] == today)]
        if not already.empty:
            return False   # already marked

    row = pd.DataFrame({
        "student_id":   [student_id],
        "student_name": [student_name],
        "timestamp":    [now.strftime("%Y-%m-%d %H:%M:%S")]
    })
    if not os.path.exists(ATTENDANCE_FILE):
        row.to_excel(ATTENDANCE_FILE, index=False)
    else:
        df_existing = pd.read_excel(ATTENDANCE_FILE)
        df_updated  = pd.concat([df_existing, row], ignore_index=True)
        df_updated.to_excel(ATTENDANCE_FILE, index=False)
    return True


def get_attendance_records(filter_date=None):
    """Return attendance DataFrame optionally filtered by date string YYYY-MM-DD."""
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=["student_id", "student_name", "timestamp"])
    df = pd.read_excel(ATTENDANCE_FILE)
    if filter_date:
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        df = df[df["_date"] == filter_date].drop(columns=["_date"])
    return df

# ─────────────────────────────────────────────
#  FACE UTILITIES
# ─────────────────────────────────────────────
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )


def load_training_data():
    """Load all saved face images and their labels."""
    samples, labels = [], []
    if not os.path.exists(PHOTOS_DIR):
        return samples, labels
    for person in os.listdir(PHOTOS_DIR):
        person_dir = os.path.join(PHOTOS_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        try:
            label = int(person)
        except ValueError:
            continue
        for fname in os.listdir(person_dir):
            fpath = os.path.join(person_dir, fname)
            img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples.append(img)
                labels.append(label)
    return samples, labels


def train_and_save_model():
    """Train recognizer on all stored photos and save to disk."""
    global model_trained
    samples, labels = load_training_data()
    if not samples:
        return False, "No training data found. Capture photos first."
    recognizer.train(samples, np.array(labels))
    recognizer.save(MODEL_FILE)
    model_trained = True
    return True, f"Model trained on {len(samples)} images from {len(set(labels))} students."


def recognize_face_in_frame(frame):
    if len(frame.shape) == 2:
        gray = frame  # already grayscale
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not model_trained:
        return None, "No Model", 0
    try:
        sid, conf = recognizer.predict(gray)
        if conf < CONFIDENCE_THRESH:
            name = get_student_name(sid)
            return sid, name, conf
    except Exception:
        pass
    return None, "Unknown", 0

# ─────────────────────────────────────────────
#  UI THEME HELPERS
# ─────────────────────────────────────────────
BG       = "#0f1117"
SURFACE  = "#1a1d27"
ACCENT   = "#4f8ef7"
ACCENT2  = "#7c5cbf"
TEXT     = "#e8eaf0"
SUBTEXT  = "#8890a4"
SUCCESS  = "#3ecf8e"
DANGER   = "#f06e6e"
BORDER   = "#2a2d3d"
FONT_H   = ("Courier New", 14, "bold")
FONT_B   = ("Courier New", 11)
FONT_SM  = ("Courier New", 9)

def style_window(win, title, w=500, h=400):
    win.title(title)
    win.geometry(f"{w}x{h}")
    win.configure(bg=BG)
    win.resizable(False, False)
    _center(win, w, h)

def _center(win, w, h):
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

def lbl(parent, text, size=11, bold=False, color=TEXT, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=("Courier New", size, weight),
                    fg=color, bg=BG, **kw)

def surf_lbl(parent, text, size=11, bold=False, color=TEXT, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=("Courier New", size, weight),
                    fg=color, bg=SURFACE, **kw)

def entry(parent, show=None, width=28):
    e = tk.Entry(parent, font=("Courier New", 11), bg=SURFACE, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", bd=6,
                 highlightthickness=1, highlightcolor=ACCENT,
                 highlightbackground=BORDER, show=show or "", width=width)
    return e

def btn(parent, text, cmd, color=ACCENT, fg="#fff", width=22):
    return tk.Button(parent, text=text, command=cmd,
                     font=("Courier New", 11, "bold"),
                     bg=color, fg=fg, activebackground=ACCENT2,
                     activeforeground="#fff", relief="flat",
                     bd=0, padx=10, pady=7, cursor="hand2", width=width)

def separator(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=5)

def status_bar(parent, var):
    return tk.Label(parent, textvariable=var, font=("Courier New", 9),
                    fg=SUBTEXT, bg=BG, anchor="w")

# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
def build_login():
    root = tk.Tk()
    style_window(root, "Attendance System — Login", 380, 320)

    tk.Frame(root, bg=BG, height=30).pack()
    lbl(root, "◈  ATTENDANCE MONITOR", size=15, bold=True, color=ACCENT).pack()
    lbl(root, "face recognition system", size=9, color=SUBTEXT).pack()
    tk.Frame(root, bg=BG, height=20).pack()

    form = tk.Frame(root, bg=BG)
    form.pack(padx=40, fill="x")

    lbl(form, "USERNAME", size=9, color=SUBTEXT).pack(anchor="w")
    u_entry = entry(form)
    u_entry.pack(fill="x", pady=(2, 10))

    lbl(form, "PASSWORD", size=9, color=SUBTEXT).pack(anchor="w")
    p_entry = entry(form, show="●")
    p_entry.pack(fill="x", pady=(2, 16))

    status_var = tk.StringVar()

    def do_login(event=None):
        if u_entry.get() == ADMIN_USERNAME and p_entry.get() == ADMIN_PASSWORD:
            root.destroy()
            build_home()
        else:
            status_var.set("✗  Invalid credentials")
            root.after(2000, lambda: status_var.set(""))

    btn(form, "  LOGIN  →", do_login, width=30).pack(fill="x")
    tk.Frame(root, bg=BG, height=8).pack()
    status_bar(root, status_var).pack(padx=40, fill="x")

    root.bind("<Return>", do_login)
    root.mainloop()

# ─────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────
def build_home():
    home = tk.Tk()
    style_window(home, "Attendance System — Home", 420, 400)

    tk.Frame(home, bg=BG, height=20).pack()
    lbl(home, "◈  DASHBOARD", size=15, bold=True, color=ACCENT).pack()
    lbl(home, "select an action below", size=9, color=SUBTEXT).pack()
    tk.Frame(home, bg=BG, height=20).pack()

    status_var = tk.StringVar()

    # ✅ Define BEFORE actions list
    def do_train_model():
        ok, msg = train_and_save_model()
        status_var.set(("✓  " if ok else "✗  ") + msg)
        home.after(4000, lambda: status_var.set(""))

    actions = [
        ("👤  Student Management",  lambda: open_student_management(home)),
        ("📷  Capture Face Photos",  lambda: open_capture_photos(home)),
        ("🧠  Train Model",          do_train_model),  # ✅ now works
        ("✅  Take Attendance",      lambda: open_take_attendance(home)),
        ("📊  Attendance Report",    lambda: open_attendance_report(home)),
        ("🚪  Logout",               lambda: (home.destroy(), build_login())),
    ]

    for text, cmd in actions:
        color = DANGER if "Logout" in text else ACCENT
        btn(home, text, cmd, color=color, width=32).pack(pady=4)

    tk.Frame(home, bg=BG, height=6).pack()
    status_bar(home, status_var).pack(padx=30, fill="x")

    home.mainloop()

    def do_train_model():
        ok, msg = train_and_save_model()
        status_var.set(("✓  " if ok else "✗  ") + msg)
        home.after(4000, lambda: status_var.set(""))

    # re-bind after definition
    actions[2] = ("🧠  Train Model", do_train_model)

    for text, cmd in actions:
        color = DANGER if "Logout" in text else ACCENT
        btn(home, text, cmd, color=color, width=32).pack(pady=4)

    tk.Frame(home, bg=BG, height=6).pack()
    status_bar(home, status_var).pack(padx=30, fill="x")

    home.mainloop()

# ─────────────────────────────────────────────
#  STUDENT MANAGEMENT
# ─────────────────────────────────────────────
def open_student_management(parent):
    win = tk.Toplevel(parent)
    style_window(win, "Student Management", 520, 560)

    tk.Frame(win, bg=BG, height=15).pack()
    lbl(win, "STUDENT MANAGEMENT", size=13, bold=True, color=ACCENT).pack()
    separator(win)

    form = tk.Frame(win, bg=BG, padx=30)
    form.pack(fill="x")

    fields = {}
    for label, key in [("Student ID", "id"), ("Full Name", "name"),
                        ("Age", "age"), ("Gender (M/F/Other)", "gender")]:
        lbl(form, label.upper(), size=9, color=SUBTEXT).pack(anchor="w", pady=(8, 0))
        e = entry(form)
        e.pack(fill="x")
        fields[key] = e

    status_var = tk.StringVar()

    def do_add():
        sid  = fields["id"].get().strip()
        name = fields["name"].get().strip()
        age  = fields["age"].get().strip()
        gen  = fields["gender"].get().strip()
        if not sid or not name:
            status_var.set("✗  ID and Name are required.")
            return
        if not sid.isdigit():
            status_var.set("✗  Student ID must be numeric.")
            return
        ok, msg = add_student_to_excel(sid, name, age, gen)
        status_var.set(("✓  " if ok else "✗  ") + msg)
        if ok:
            for e in fields.values():
                e.delete(0, "end")
            refresh_table()

    separator(win)
    btn(win, "➕  Add Student", do_add, width=30).pack(pady=8)
    tk.Label(win, textvariable=status_var, font=FONT_SM, fg=SUCCESS, bg=BG).pack()
    separator(win)

    # ── student table ──
    lbl(win, "REGISTERED STUDENTS", size=9, color=SUBTEXT).pack(anchor="w", padx=30)
    frame_tbl = tk.Frame(win, bg=BG, padx=20)
    frame_tbl.pack(fill="both", expand=True, pady=5)

    cols = ("ID", "Name", "Age", "Gender")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Custom.Treeview",
                    background=SURFACE, foreground=TEXT,
                    fieldbackground=SURFACE, rowheight=22,
                    font=("Courier New", 9))
    style.configure("Custom.Treeview.Heading",
                    background=BORDER, foreground=ACCENT,
                    font=("Courier New", 9, "bold"))

    tree = ttk.Treeview(frame_tbl, columns=cols, show="headings",
                        style="Custom.Treeview", height=6)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=100, anchor="center")
    tree.pack(fill="both", expand=True)

    def refresh_table():
        tree.delete(*tree.get_children())
        for _, row in get_all_students().iterrows():
            tree.insert("", "end", values=(
                row["student_id"], row["student_name"],
                row.get("age", ""), row.get("gender", "")
            ))

    refresh_table()

# ─────────────────────────────────────────────
#  CAPTURE PHOTOS
# ─────────────────────────────────────────────
def open_capture_photos(parent):
    win = tk.Toplevel(parent)
    style_window(win, "Capture Face Photos", 400, 260)

    tk.Frame(win, bg=BG, height=15).pack()
    lbl(win, "CAPTURE FACE PHOTOS", size=13, bold=True, color=ACCENT).pack()
    lbl(win, f"captures {PHOTOS_PER_STUDENT} grayscale face samples", size=9, color=SUBTEXT).pack()
    separator(win)

    form = tk.Frame(win, bg=BG, padx=40)
    form.pack(fill="x")
    lbl(form, "STUDENT ID", size=9, color=SUBTEXT).pack(anchor="w", pady=(8, 0))
    sid_entry = entry(form)
    sid_entry.pack(fill="x")

    status_var = tk.StringVar()
    progress_var = tk.IntVar()

    def do_capture():
        sid = sid_entry.get().strip()
        if not sid.isdigit():
            status_var.set("✗  Enter a valid numeric Student ID.")
            return
        status_var.set("📷  Capturing... press Q to stop early.")
        win.update()

        cap   = cv2.VideoCapture(0)
        count = 0
        student_dir = os.path.join(PHOTOS_DIR, sid)
        os.makedirs(student_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), roi)
                count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 142, 247), 2)
                cv2.putText(frame, f"Captured: {count}/{PHOTOS_PER_STUDENT}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (79, 142, 247), 2)
            cv2.imshow("Capturing Faces — Press Q to stop", frame)
            if cv2.waitKey(50) & 0xFF == ord("q") or count >= PHOTOS_PER_STUDENT:
                break

        cap.release()
        cv2.destroyAllWindows()
        status_var.set(f"✓  Saved {count} photos for student ID {sid}. Now train the model.")

    separator(win)
    btn(win, "📷  Start Capture", do_capture, width=30).pack(pady=10)
    tk.Label(win, textvariable=status_var, font=FONT_SM, fg=SUCCESS, bg=BG,
             wraplength=340).pack()

# ─────────────────────────────────────────────
#  TAKE ATTENDANCE
# ─────────────────────────────────────────────
def open_take_attendance(parent):
    if not model_trained:
        messagebox.showwarning("No Model",
            "No trained model found.\nPlease train the model first.")
        return

    win = tk.Toplevel(parent)
    style_window(win, "Take Attendance", 460, 340)

    tk.Frame(win, bg=BG, height=15).pack()
    lbl(win, "TAKE ATTENDANCE", size=13, bold=True, color=ACCENT).pack()
    lbl(win, f"today: {date.today().isoformat()}", size=9, color=SUBTEXT).pack()
    separator(win)

    log_frame = tk.Frame(win, bg=SURFACE, relief="flat", bd=0)
    log_frame.pack(padx=20, fill="both", expand=True)

    log = tk.Text(log_frame, font=("Courier New", 9), bg=SURFACE, fg=TEXT,
                  relief="flat", state="disabled", height=10)
    log.pack(fill="both", expand=True, padx=8, pady=8)

    status_var = tk.StringVar(value="Press Start to begin...")
    status_bar(win, status_var).pack(padx=20, fill="x")

    running = [False]

    def log_msg(msg, color=TEXT):
        log.configure(state="normal")
        log.insert("end", msg + "\n")
        log.see("end")
        log.configure(state="disabled")

    def do_start():
        if not model_trained:
            return
        running[0] = True
        btn_start.configure(state="disabled")
        btn_stop.configure(state="normal")
        status_var.set("🟢  Camera active — recognizing faces...")

        cap = cv2.VideoCapture(0)
        marked_today = set()  # session-level duplicate prevention

        def loop():
            if not running[0]:
                cap.release()
                cv2.destroyAllWindows()
                return

            ret, frame = cap.read()
            if not ret:
                win.after(100, loop)
                return

            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                roi  = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                sid, name, conf = recognize_face_in_frame(roi)

                color_cv = (62, 207, 142) if name != "Unknown" else (240, 110, 110)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_cv, 2)
                cv2.putText(frame, f"{name} ({conf:.0f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_cv, 2)

                if sid and sid not in marked_today:
                    ok = mark_attendance(sid, name)
                    marked_today.add(sid)
                    ts = datetime.now().strftime("%H:%M:%S")
                    if ok:
                        log_msg(f"[{ts}] ✓  {name} (ID {sid}) — attendance marked")
                    else:
                        log_msg(f"[{ts}] ⚠  {name} (ID {sid}) — already marked today")

            cv2.imshow("Taking Attendance — Press Q to stop", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                running[0] = False

            win.after(10, loop)

        win.after(100, loop)

    def do_stop():
        running[0] = False
        btn_start.configure(state="normal")
        btn_stop.configure(state="disabled")
        status_var.set("⏹  Stopped.")

    separator(win)
    btn_row = tk.Frame(win, bg=BG)
    btn_row.pack(pady=6)
    btn_start = btn(btn_row, "▶  Start", do_start, color=SUCCESS, width=14)
    btn_start.pack(side="left", padx=6)
    btn_stop  = btn(btn_row, "⏹  Stop",  do_stop,  color=DANGER,  width=14)
    btn_stop.pack(side="left", padx=6)
    btn_stop.configure(state="disabled")

    win.protocol("WM_DELETE_WINDOW", lambda: (do_stop(), win.destroy()))

# ─────────────────────────────────────────────
#  ATTENDANCE REPORT
# ─────────────────────────────────────────────
def open_attendance_report(parent):
    win = tk.Toplevel(parent)
    style_window(win, "Attendance Report", 600, 500)

    tk.Frame(win, bg=BG, height=15).pack()
    lbl(win, "ATTENDANCE REPORT", size=13, bold=True, color=ACCENT).pack()
    separator(win)

    # filter bar
    filter_row = tk.Frame(win, bg=BG)
    filter_row.pack(padx=20, fill="x", pady=4)
    lbl(filter_row, "Filter by date (YYYY-MM-DD):", size=9, color=SUBTEXT).pack(side="left")
    date_entry = entry(filter_row, width=14)
    date_entry.insert(0, date.today().isoformat())
    date_entry.pack(side="left", padx=8)
    btn(filter_row, "🔍 Filter", lambda: refresh(date_entry.get().strip()),
        width=10).pack(side="left", padx=4)
    btn(filter_row, "🔄 All", lambda: refresh(""), width=8).pack(side="left")

    separator(win)

    # table
    frame_tbl = tk.Frame(win, bg=BG, padx=16)
    frame_tbl.pack(fill="both", expand=True)

    style = ttk.Style()
    style.configure("Rep.Treeview",
                    background=SURFACE, foreground=TEXT,
                    fieldbackground=SURFACE, rowheight=22,
                    font=("Courier New", 9))
    style.configure("Rep.Treeview.Heading",
                    background=BORDER, foreground=ACCENT,
                    font=("Courier New", 9, "bold"))

    cols = ("ID", "Name", "Timestamp")
    tree = ttk.Treeview(frame_tbl, columns=cols, show="headings",
                        style="Rep.Treeview")
    for c, w in zip(cols, [60, 200, 180]):
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor="center")

    sb = ttk.Scrollbar(frame_tbl, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    count_var = tk.StringVar()
    tk.Label(win, textvariable=count_var, font=FONT_SM, fg=SUBTEXT, bg=BG).pack(pady=4)

    def refresh(filter_date=""):
        tree.delete(*tree.get_children())
        df = get_attendance_records(filter_date if filter_date else None)
        for _, row in df.iterrows():
            tree.insert("", "end", values=(
                row["student_id"], row["student_name"], row["timestamp"]
            ))
        count_var.set(f"{len(df)} record(s) shown")

    refresh(date.today().isoformat())

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        _load_model_if_exists()
        build_login()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
