# Helmet Detection System

This project is a complete Streamlit UI for helmet detection using the trained YOLO26n weights in `best_model/best.pt`.

## What is included

- `app.py` - Streamlit website with sidebar navigation.
- `best_model/best.pt` - trained YOLO model.
- `data.yaml` - YOLO dataset configuration.
- `training_data/` - train and validation images/labels used for analytics and retraining.
- `Train.ipynb`, `Test.ipynb`, `test_model.py`, `result.py`, `split_data.py` - training and testing support files.

## Website sections

- Project Overview
- Test Detection
- Dataset Analytics
- Evaluation Metrics

## How to run

Open a terminal in this folder and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you use macOS/Linux, activate the environment with:

```bash
source venv/bin/activate
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

## How to test detection

1. Start the app with `streamlit run app.py`.
2. Select `Test Detection` from the sidebar.
3. Choose `Upload Image` or `Live Capturing`.
4. For upload, select a `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` image and click `Detect Uploaded Image`.
5. For live capture, click `Start Live Capturing`. OpenCV opens the default camera in a separate desktop window, runs live detection, and closes when you press `Q` in that OpenCV window.
6. The app displays the annotated image, predicted class, confidence, and bounding box values.

There is no default image fallback. You must upload an image or use live camera capture.

## Retraining command

You can retrain or validate with Ultralytics commands such as:

```bash
yolo detect train model=yolo26n.pt data=data.yaml imgsz=640 epochs=50
yolo detect val model=best_model/best.pt data=data.yaml imgsz=640
```
