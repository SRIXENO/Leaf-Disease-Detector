from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tensorflow.keras.models import load_model, Model
import numpy as np, uvicorn, os, uuid, cv2, tensorflow as tf, pandas as pd
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from deep_translator import GoogleTranslator

import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime

from gradcam_utils import get_gradcam_heatmap, get_simple_gradcam_heatmap, overlay_gradcam

load_dotenv("API.env")
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
print("Loaded OpenAI key:", (OPENAI_API_KEY[:8] + "...") if OPENAI_API_KEY else "MISSING")
print("Loaded OpenWeather key:", ("Present" if OPENWEATHER_API_KEY else "MISSING"))

app = FastAPI(title="Plant Disease Detection with Remedies")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "C:/Users/mrvss/Downloads/LeafDiseaseProject/dataset/model.h5"  
try:
    model = load_model(MODEL_PATH, compile=False)
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    _ = model(dummy_input, training=False)
    if isinstance(model, tf.keras.Sequential):
        model = Model(inputs=model.inputs, outputs=model.outputs)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

remedy_db = None
if os.path.exists("remedies.csv"):
    try:
        remedy_db = pd.read_csv("remedies.csv")
    except Exception as e:
        print("Could not read remedies.csv:", e)

class_names = [
    "Alstonia Scholaris___diseased",
    "Alstonia Scholaris___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Arjun___diseased",
    "Arjun___healthy",
    "Bael___diseased",
    "Basil___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Chinar ___diseased",
    "Chinar ___healthy",
]

def get_severity(confidence: float) -> str:
    if confidence > 85:
        return "Severe"
    elif confidence > 60:
        return "Moderate"
    return "Mild"

def get_ai_remedy(disease_name: str, city: str = None, weather_info: dict = None) -> str:
    """
    Get AI-generated remedy from GPT, considering disease type, city, and weather.
    """
    if not disease_name or "healthy" in disease_name.lower():
        return "🌿 Your plant appears healthy! No treatment required.\n\n✅ *Precaution Tips:*\n- Water early in the morning.\n- Maintain good soil drainage.\n- Keep an eye out for pests weekly.\n\n🌾 *Fertilizer Advice:*\nUse organic compost or balanced NPK (10-10-10) monthly."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

       
        user_prompt = f"""
        You are an expert agricultural assistant helping a farmer in {city or "India"}.
        The detected plant disease is: {disease_name}.
        Current weather conditions: {weather_info or "Unknown"}.

        Please provide:
        1️⃣  Short, clear remedies in bullet points.
        2️⃣  Precautionary measures to prevent recurrence.
        3️⃣  Recommended fertilizers (organic and chemical).
        4️⃣  Tailor advice to the given location & weather.
        Keep it concise and farmer-friendly.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an agricultural expert providing simple, location-based remedies."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        return text

    except Exception as e:
        print("⚠️ OpenAI failed:", e)

        if remedy_db is not None:
            match = remedy_db.loc[remedy_db["Disease"] == disease_name]
            if not match.empty:
                return match["Remedy"].values[0]

        return "No AI solution available for this disease."

def translate_text(text: str, lang="en") -> str:

    try:
        if lang == "en":
            return text
        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text

def get_city_from_coordinates(lat: float, lon: float) -> str:
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        res = requests.get(url, headers={"User-Agent": "PlantAI-Pro"}, timeout=8)
        data = res.json()
        addr = data.get("address", {})
        for k in ["city", "town", "village", "county", "state"]:
            if addr.get(k):
                return addr.get(k)
        return "Unknown"
    except Exception as e:
        print("Reverse geocode failed:", e)
        return "Unknown"

def get_weather_risk_by_coords(lat: float = None, lon: float = None, city: str = None):
    """Return dict: {city, temperature, humidity, risk}"""
    try:
        if lat is not None and lon is not None:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        else:
            if not city: city = "Chennai"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=8).json()
        main = res.get("main", {})
        temp = main.get("temp")
        humidity = main.get("humidity")
        risk = "Low"
        if humidity is not None:
            if humidity > 80 or (temp is not None and temp > 34):
                risk = "High"
            elif humidity > 60:
                risk = "Moderate"
        return {"city": city or res.get("name", "Unknown"), "temperature": temp, "humidity": humidity, "risk": risk}
    except Exception as e:
        print("Weather API failed:", e)
        return {"city": city or "Unknown", "temperature": None, "humidity": None, "risk": "Unknown"}

def generate_pdf_report(prediction, confidence, severity, remedy, gradcam_path, weather_info, filename):
    pdf_path = f"static/reports/{filename}.pdf"
    os.makedirs("static/reports", exist_ok=True)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setTitle("Plant Health Report")

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "🌿 Plant Disease Detection Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, 780, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.drawString(50, 765, f"Disease: {prediction}")
    c.drawString(50, 750, f"Confidence: {confidence}%")
    c.drawString(50, 735, f"Severity: {severity}")

    c.drawString(50, 715, f"Weather Risk: {weather_info.get('risk')} (Humidity: {weather_info.get('humidity')}%)")
    c.drawString(50, 700, f"Location: {weather_info.get('city')}")

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 675, "Recommended Remedy:")
    c.setFont("Helvetica", 10)
    text = c.beginText(50, 655)
    for line in remedy.split("\n"):
        text.textLine(line)
    c.drawText(text)

    if gradcam_path and os.path.exists(gradcam_path):
        try:
            c.drawImage(gradcam_path, 50, 380, width=380, height=250)
        except Exception as e:
            print("Could not place image in PDF:", e)

    c.showPage()
    c.save()
    return pdf_path

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "OPENWEATHER_KEY": OPENWEATHER_API_KEY 
    })

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = "en",
    lat: float = None,
    lon: float = None,
    city: str = None
):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    try:
        file_path = f"static/uploads/{uuid.uuid4().hex}_{file.filename}"
        os.makedirs("static/uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        image = Image.open(file_path).convert("RGB")
        image = image.resize((256, 256))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "Unknown"
        confidence = float(np.max(predictions)) * 100.0
        severity = get_severity(confidence)

        weather_info = {"city": "Unknown", "temperature": None, "humidity": None, "risk": "Unknown"}
        try:
            if lat and lon:
                detected_city = get_city_from_coordinates(lat, lon)
                weather_info = get_weather_risk_by_coords(lat=lat, lon=lon, city=detected_city)
            elif city:
                weather_info = get_weather_risk_by_coords(city=city)
            else:
                weather_info = get_weather_risk_by_coords(city="Chennai")
        except Exception as e:
            print("Weather info fetch failed:", e)

        remedy = get_ai_remedy(predicted_class, city=weather_info.get("city"), weather_info=weather_info)
        remedy_translated = translate_text(remedy, lang=lang)

        top_indices = predictions[0].argsort()[-3:][::-1]
        suggestions = [
            (class_names[int(i)] if int(i) < len(class_names) else "Unknown", float(predictions[0][int(i)]) * 100.0)
            for i in top_indices
        ]

        gradcam_path = None
        try:
            conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
            if conv_layers:
                last_conv_layer_name = conv_layers[-1].name
                try:
                    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
                except Exception as e:
                    print("Grad-CAM gradients failed, falling back to simple:", e)
                    heatmap = get_simple_gradcam_heatmap(model, img_array, last_conv_layer_name)
                gradcam_img = overlay_gradcam(file_path, heatmap)
                os.makedirs("static/gradcam", exist_ok=True)
                gradcam_path = f"static/gradcam/{uuid.uuid4().hex}.jpg"
                cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print("Grad-CAM generation failed:", e)

        pdf_path = None
        try:
            pdf_name = uuid.uuid4().hex
            pdf_path = generate_pdf_report(
                predicted_class,
                round(confidence, 2),
                severity,
                remedy_translated,
                gradcam_path,
                weather_info,
                pdf_name,
            )
        except Exception as e:
            print("PDF generation failed:", e)

        response = {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "severity": severity,
            "remedy": remedy,
            "remedy_translated": remedy_translated,
            "suggestions": suggestions,
            "weather": weather_info,
            "gradcam_image": ("/" + gradcam_path) if gradcam_path else None,
            "report_pdf": ("/" + pdf_path) if pdf_path else None,
        }
        return JSONResponse(response)

    except Exception as e:
        print("Predict error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/static/reports/{filename}")
async def get_report(filename: str):
    path = os.path.join("static", "reports", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="application/pdf", filename=filename)
    return JSONResponse({"error":"Not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")