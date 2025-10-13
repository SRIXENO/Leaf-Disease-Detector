# AI-Leaf-Disease-Detector
An AI-based web app for detecting plant leaf diseases with weather-based analysis and bilingual voice support (English, Hindi &amp; Tamil).
---

## 🚀 Features

- 🌱 **AI Disease Detection:** Upload a leaf image to identify diseases instantly.  
- 🌤️ **Weather-Aware Risk:** Uses OpenWeather API to assess environmental risk levels.  
- 🗣️ **Voice Assistant:** Reads remedies aloud (supports English & Tamil).  
- 🧠 **Explainable AI:** Displays Grad-CAM visualization of infected regions.  
- 💾 **Downloadable Report:** Generates a professional treatment report.  

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask (Python) |
| AI Model | TensorFlow / Keras CNN |
| API | OpenWeatherMap |
| Deployment | GitHub  |

---
# requirements.txt

Flask==3.0.3
tensorflow==2.15.0
numpy==1.26.4
opencv-python==4.9.0.80
Pillow==10.2.0
python-dotenv==1.0.1
requests==2.31.0
gunicorn==22.0.0
werkzeug==3.0.3

📷 Project Structure
AI-Leaf-Disease-Detector/
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── model/
│   └── model.h5
│
├── app.py
├── requirements.txt
├── README.md
└── .env
