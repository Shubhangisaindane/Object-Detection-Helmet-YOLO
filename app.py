from __future__ import annotations

import tempfile
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import yaml
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model" / "best.pt"
DATA_YAML_PATH = BASE_DIR / "data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE = 0.45
DEFAULT_LINE_WIDTH = 2


st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp { overflow-x: hidden; }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        width: 100%;
        max-width: min(1120px, calc(100vw - 2rem));
    }
    section[data-testid="stSidebar"] { min-width: 270px; }
    h1, h2, h3 {
        max-width: 100%;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: normal !important;
        line-height: 1.18 !important;
        letter-spacing: 0 !important;
    }
    h1 { font-size: clamp(1.45rem, 2.5vw, 2rem) !important; }
    h2 { font-size: clamp(1.2rem, 2vw, 1.45rem) !important; }
    h3 { font-size: clamp(1rem, 1.5vw, 1.2rem) !important; }
    .muted {
        color: #667085;
        font-size: .98rem;
        line-height: 1.5;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    .status-good { color: #047857; font-weight: 700; }
    .status-bad { color: #b42318; font-weight: 700; }
    p, li, div, span {
        max-width: 100%;
        overflow-wrap: break-word;
        word-break: normal;
    }
    div[data-testid="column"] { min-width: 0; }
    button, label, textarea, input { max-width: 100%; }
    [data-testid="stDataFrame"] { width: 100%; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
    }
    /* Equal image heights */
[data-testid="stImage"] img {
    width: 100% !important;
    height: 450px !important;
    object-fit: contain !important;
    border-radius: 10px;
}

/* Make upload area compact */
[data-testid="stFileUploader"] {
    width: 100%;
}

/* Full width button */
.stButton button {
    width: 100%;
}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return YOLO(str(path))


@st.cache_data
def load_dataset_config(path: str) -> dict:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {"nc": 0, "names": {}}
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


@st.cache_data
def collect_dataset_stats(base_dir: str, data_yaml_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(base_dir)
    config = load_dataset_config(data_yaml_path)
    dataset_root = Path(config.get("path", base / "training_data"))
    if not dataset_root.is_absolute():
        dataset_root = base / dataset_root

    rows = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_count = count_files(image_dir, IMAGE_EXTENSIONS)
        label_count = count_files(label_dir, {".txt"})
        if image_count or label_count:
            rows.append({"Split": split.title(), "Images": image_count, "Labels": label_count})

    split_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Split", "Images", "Labels"])

    names = normalize_names(config.get("names", {}))
    class_counts = {name: 0 for name in names.values()} or {"Helmet": 0}
    labels_root = dataset_root / "labels"
    for label_file in labels_root.rglob("*.txt") if labels_root.exists() else []:
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                class_name = names.get(int(parts[0]), f"Class {parts[0]}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    class_df = pd.DataFrame(
        [{"Class": class_name, "Boxes": count} for class_name, count in class_counts.items()]
    )
    return split_df, class_df


def count_files(path: Path, extensions: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix.lower() in extensions)


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def get_inference_options() -> dict:
    return {
        "confidence": DEFAULT_CONFIDENCE,
        "show_boxes": True,
        "show_labels": True,
        "line_width": DEFAULT_LINE_WIDTH,
    }


def image_to_temp_file(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name, format="PNG")
        return temp_file.name


def run_prediction(model: YOLO, image: Image.Image, options: dict) -> tuple[Image.Image, pd.DataFrame]:
    image_path = image_to_temp_file(image)

    try:
        results = model.predict(
            source=image_path,
            conf=options["confidence"],
            verbose=False
        )

        result = results[0]

        annotated = Image.fromarray(result.plot()[:, :, ::-1])

        rows = []

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(result.names.get(cls_id, f"Class {cls_id}"))
            confidence = float(box.conf[0].item())

            xmin, ymin, xmax, ymax = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            rows.append(
                {
                    "Class": class_name,
                    "Confidence": round(confidence * 100, 2),
                    "Xmin": round(xmin, 1),
                    "Ymin": round(ymin, 1),
                    "Xmax": round(xmax, 1),
                    "Ymax": round(ymax, 1),
                }
            )

        return annotated, pd.DataFrame(rows)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

def show_prediction_summary(predictions: pd.DataFrame, empty_message: str = "No helmet detected.") -> None:
    if predictions.empty:
        st.error(empty_message)
        return

    best_row = predictions.sort_values("Confidence", ascending=False).iloc[0]
    st.success(f"Helmet detected with {best_row['Confidence']:.2f}% confidence.")
    st.dataframe(predictions, hide_index=True, use_container_width=True)

def sidebar() -> str:
    st.sidebar.title("Helmet Detection")
    st.sidebar.caption("YOLO26n + Streamlit")
    page = st.sidebar.radio(
        "Navigation",
        ["Project Overview", "Test Detection", "Dataset Analytics", "Evaluation Metrics"],
    )
    st.sidebar.divider()
    st.sidebar.write("Model status")
    if MODEL_PATH.exists():
        st.sidebar.success("best_model/best.pt found")
    else:
        st.sidebar.error("Model missing")
    st.sidebar.caption("Upload an image from Test Detection.")
    return page


def project_overview() -> None:
    st.title("Smart Helmet Detection System")
    st.markdown('<div class="muted">Detect whether a rider/person is wearing a helmet using YOLO26n.</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.subheader("Project Objective")
        st.write(
            "The objective of this project is to detect whether a person is wearing a helmet or not "
            "using a trained YOLO26n model. The website provides image-based testing, dataset analytics, "
            "evaluation summaries, and live camera-based detection."
        )
        st.subheader("Applications")
        st.write(
            "- Traffic Surveillance\n"
            "- Rider Safety Monitoring\n"
            "- Accident Prevention\n"
            "- Smart City Applications\n"
            "- Automated Safety Compliance"
        )
    with col2:
        st.subheader("Technology Stack")
        stack = pd.DataFrame(
            {
                "Component": ["Language", "Computer Vision", "Detection Model", "UI", "Annotation", "Tracking"],
                "Tool": ["Python", "OpenCV", "YOLO26n / Ultralytics", "Streamlit", "Label Studio", "MLflow"],
            }
        )
        st.dataframe(stack, hide_index=True, use_container_width=True)

    st.subheader("Workflow")
    st.info("Upload image -> preprocess image -> run YOLO prediction -> draw bounding boxes -> show result and confidence")


def test_detection(model: YOLO, options: dict) -> None:
    st.title("Test Detection")

    st.markdown(
        '<div class="muted">Upload an image and run the YOLO helmet detector.</div>',
        unsafe_allow_html=True
    )

    upload_col, button_col = st.columns([1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload rider image",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

    with button_col:
        st.write("")
        st.write("")
        detect_btn = st.button(
            "Detect Helmet",
            type="primary",
            use_container_width=True
        )

    if uploaded_file is None:
        st.info("Upload an image to enable detection.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    if not detect_btn:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with right_col:
            st.subheader("Detection Result")
            st.info("Press Detect Helmet to run the model.")

        return

    with st.spinner("Running YOLO inference..."):
        annotated, predictions = run_prediction(
            model,
            image,
            options
        )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right_col:
        st.subheader("Detection Result")
        st.image(annotated, use_container_width=True)

    st.markdown("---")

    if predictions.empty:
        st.error("No helmet detected in this image.")
    else:
        best_row = predictions.sort_values(
            "Confidence",
            ascending=False
        ).iloc[0]

        st.success(
            f"Helmet detected with {best_row['Confidence']:.2f}% confidence."
        )

        st.dataframe(
            predictions,
            hide_index=True,
            use_container_width=True
        )


def dataset_analytics() -> None:
    st.title("Dataset Analytics")
    split_df, class_df = collect_dataset_stats(str(BASE_DIR), str(DATA_YAML_PATH))

    total_images = int(split_df["Images"].sum()) if not split_df.empty else 0
    total_labels = int(split_df["Labels"].sum()) if not split_df.empty else 0
    total_boxes = int(class_df["Boxes"].sum()) if not class_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_images)
    col2.metric("Label Files", total_labels)
    col3.metric("Annotated Objects", total_boxes)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Images by Split")
        if split_df.empty:
            st.warning("No dataset folders were found.")
        else:
            st.bar_chart(split_df.set_index("Split")[["Images", "Labels"]])
            st.dataframe(split_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Class Distribution")
        if class_df.empty:
            st.warning("No class labels were found.")
        else:
            st.bar_chart(class_df.set_index("Class")["Boxes"])
            st.dataframe(class_df, hide_index=True, use_container_width=True)


def evaluation_metrics(model: YOLO) -> None:
    st.title("Evaluation Metrics")

    st.markdown(
        '<div class="muted">View model details and run validation on the bundled YOLO dataset.</div>',
        unsafe_allow_html=True
    )

    config = load_dataset_config(str(DATA_YAML_PATH))
    names = normalize_names(config.get("names", {}))

    model_names = getattr(model, "names", {})
    model_classes = (
        ", ".join(str(v) for v in model_names.values())
        if isinstance(model_names, dict)
        else str(model_names)
    )

    # ==================================================
    # MODEL SUMMARY
    # ==================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "YOLO26n")
    col2.metric(
        "Classes",
        len(model_names) if isinstance(model_names, dict)
        else config.get("nc", 0)
    )
    col3.metric("Confidence", f"{DEFAULT_CONFIDENCE:.2f}")
    col4.metric("Image Size", "640")

    st.divider()

    # ==================================================
    # MODEL DETAILS
    # ==================================================
    st.subheader("Model Details")

    details = pd.DataFrame(
        [
            {
                "Property": "Weights",
                "Value": str(MODEL_PATH.relative_to(BASE_DIR))
            },
            {
                "Property": "Dataset Config",
                "Value": str(DATA_YAML_PATH.relative_to(BASE_DIR))
            },
            {
                "Property": "Dataset Classes",
                "Value": ", ".join(names.values()) or "Not Available"
            },
            {
                "Property": "Model Classes",
                "Value": model_classes or "Not Available"
            },
            {
                "Property": "Framework",
                "Value": "Ultralytics YOLO"
            },
        ]
    )

    st.dataframe(
        details,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ==================================================
    # VALIDATION
    # ==================================================
    st.subheader("Run Evaluation")

    st.write(
        "Click the button below to evaluate the model using the validation dataset."
    )

    if st.button("Run Validation", type="primary"):

        try:
            with st.spinner("Running validation..."):

                metrics = model.val(
                    data=str(DATA_YAML_PATH),
                    imgsz=640,
                    conf=DEFAULT_CONFIDENCE,
                    verbose=False
                )

            box_metrics = getattr(metrics, "box", None)

            metric_rows = pd.DataFrame(
                [
                    {
                        "Metric": "Precision",
                        "Value": metric_value(box_metrics, "mp")
                    },
                    {
                        "Metric": "Recall",
                        "Value": metric_value(box_metrics, "mr")
                    },
                    {
                        "Metric": "mAP50",
                        "Value": metric_value(box_metrics, "map50")
                    },
                    {
                        "Metric": "mAP50-95",
                        "Value": metric_value(box_metrics, "map")
                    },
                ]
            )

            # Display table
            display_rows = metric_rows.copy()

            display_rows["Value"] = display_rows["Value"].apply(
                lambda x: f"{x * 100:.2f}%"
                if pd.notna(x)
                else "N/A"
            )

            st.success("Validation completed successfully.")

            st.dataframe(
                display_rows,
                hide_index=True,
                use_container_width=True
            )

            # Display metrics cards
            valid_metrics = metric_rows.set_index("Metric")["Value"]

            m1, m2, m3, m4 = st.columns(4)

            if pd.notna(valid_metrics.get("Precision")):
                m1.metric(
                    "Precision",
                    f"{valid_metrics['Precision'] * 100:.2f}%"
                )

            if pd.notna(valid_metrics.get("Recall")):
                m2.metric(
                    "Recall",
                    f"{valid_metrics['Recall'] * 100:.2f}%"
                )

            if pd.notna(valid_metrics.get("mAP50")):
                m3.metric(
                    "mAP50",
                    f"{valid_metrics['mAP50'] * 100:.2f}%"
                )

            if pd.notna(valid_metrics.get("mAP50-95")):
                m4.metric(
                    "mAP50-95",
                    f"{valid_metrics['mAP50-95'] * 100:.2f}%"
                )

            # Chart
            chart_rows = metric_rows.dropna()

            if not chart_rows.empty:
                st.subheader("Metrics Chart")

                st.bar_chart(
                    chart_rows.set_index("Metric")["Value"]
                )

        except Exception as e:
            st.error(f"Validation failed: {e}")

    st.divider()

    # ==================================================
    # COMMAND
    # ==================================================
    st.subheader("Equivalent YOLO Command")

    st.code(
        "yolo val model=best_model/best.pt data=data.yaml imgsz=640",
        language="bash"
    )

def metric_value(box_metrics: object, attribute: str) -> float | None:
    if box_metrics is None or not hasattr(box_metrics, attribute):
        return None
    value = getattr(box_metrics, attribute)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def main() -> None:
    get_inference_options()
    page = sidebar()

    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    if page == "Project Overview":
        project_overview()
    elif page == "Test Detection":
        test_detection(model, get_inference_options())
    elif page == "Dataset Analytics":
        dataset_analytics()
    elif page == "Evaluation Metrics":
        evaluation_metrics(model)


if __name__ == "__main__":
    main()
