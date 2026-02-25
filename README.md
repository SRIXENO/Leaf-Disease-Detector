# Leaf-Disease-Detector

AI-powered leaf disease detection web app with:
- TensorFlow model inference
- weather-aware risk hints
- AI-generated remedies
- multilingual remedy translation
- Grad-CAM visualization
- downloadable PDF report

## Stack
- Backend: FastAPI (Python)
- Model: TensorFlow / Keras (`dataset/model.h5`)
- Frontend: HTML, CSS, JavaScript
- APIs: OpenWeatherMap, OpenAI

## Quick Start (Windows)
1. Clone the repository.
2. Open project folder:
```powershell
cd "Leaf-Disease-Detector"
```
3. Install dependencies:
```powershell
pip install fastapi uvicorn tensorflow keras numpy opencv-python pillow jinja2 python-multipart python-dotenv openai deep-translator reportlab requests pandas
```
4. Add API keys in `dataset/.env` or `dataset/API.env`:
- `OPENAI_API_KEY=...`
- `OPENWEATHER_API_KEY=...`
5. Start app:
```powershell
.\start_project.bat
```
6. Open:
- `http://127.0.0.1:8080`

## Manual Run (without BAT)
```powershell
cd dataset
python app.py
```

## Project Structure
```text
Leaf-Disease-Detector/
|-- start_project.bat
|-- dataset/
|   |-- app.py
|   |-- model.h5
|   |-- Grad_cam_CNN.py
|   |-- gradcam_utils.py
|   |-- train model.py
|   |-- templates/
|   |-- static/
|   `-- reports.json
`-- README.md
```

## API
- `GET /` -> web UI
- `POST /predict` -> prediction + remedy + weather + Grad-CAM + PDF link

## Notes
- `model.h5` is tracked with Git LFS.
- Large generated folders are excluded from Git.

## Disclaimer
- The image dataset is not included in this repository.
- The following folders are intentionally excluded from GitHub:
  - `dataset/train/`
  - `dataset/test/`
  - `dataset/valid/`
  - `dataset/gradcam_outputs/`
