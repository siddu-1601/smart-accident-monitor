# Smart Accident Monitor

Smart Accident Monitor is a Deep Learning based system that detects accident severity from video footage and classifies it as **MINOR, MAJOR, or SEVERE**. This project works as a **local demo and academic project** and is designed for future cloud-based deployment.

---

## 🔧 Technologies Used

- Python  
- FastAPI  
- PyTorch (ResNet18)  
- OpenCV  
- HTML + Tailwind CSS  
- Supabase (future use)  
- Render (future deployment)  

---

## ✅ Current Features

- Video-based accident severity prediction  
- Deep Learning model using ResNet18  
- Severity classification: MINOR, MAJOR, SEVERE  
- Local terminal-based execution  
- Optional frontend for uploading video  
- Alert trigger logic for SEVERE cases  
- Ready for future cloud deployment  

---

## 🎯 Project Working & Motto

**Working:**  
The system takes a road accident video as input, extracts key frames, and passes them through a trained ResNet18 deep learning model to predict the accident severity as MINOR, MAJOR, or SEVERE. The prediction is displayed instantly in the terminal and can also be visualized through the frontend. For SEVERE cases, the system is designed to trigger an emergency alert mechanism.

**Motto:**  
To enable faster accident severity detection and support quicker emergency response using intelligent video analysis.

---

## ▶️ How to Run the Project (Local / Codespaces)

### 1. Clone the Repository

```bash
git clone https://github.com/siddu-1601/smart-accident-monitor.git
cd smart-accident-monitor

```

### 2. Install the Required Packages

```bash
pip install -r requirements.txt

```

### 3. Run the Model on a Video

```bash
python -m backend.debug_severity FireAcc_1.mp4

```

---

## 👤 Author

**Venkat Siddarth**  
Computer Science Engineering Student  
GitHub: https://github.com/siddu-1601  
Email: vsiddarth401@gmail.com  
Resume: https://drive.google.com/file/d/1vQ6WKIC24iw8FD9YT-Ndkh2-p26QS6IO/view  


