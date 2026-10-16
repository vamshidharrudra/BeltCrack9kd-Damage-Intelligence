
import os
import re
import json
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image
from matplotlib.patches import Polygon
from dotenv import load_dotenv
from groq import Groq
from streamlit_drawable_canvas import st_canvas


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name == "BeltCrack9kd_Damage_Intelligence" else APP_DIR
DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

PREDICTION_PATH = ROOT / "audit_outputs" / "stage_3_27_final_inference_predictions.csv"
MANIFEST_PATH = ROOT / "canonical_dataset" / "canonical_manifest.csv"
CHECKPOINT_PATH = ROOT / "models" / "segmentation" / "beltcrack9kd_cpu_pilot_best.pt"

load_dotenv(ROOT / ".env")

IMAGE_SIZE = 256
MASK_THRESHOLD = 0.10
MAX_VIDEO_FRAMES = 60
DEVICE = torch.device("cpu")

torch.set_num_threads(2)

st.set_page_config(
    page_title="BeltCrack9kd Damage Intelligence",
    page_icon="🔎",
    layout="wide",
)

st.title("BeltCrack9kd Damage Intelligence")
st.caption("Sequential dashboard | Belt-only image/video analysis | Groq AI")
st.warning(
    "Experimental segmentation model. Predicted crack-like pixels are "
    "not verified physical damage."
)


# ============================================================
# SEGMENTATION MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(3, 16)
        self.e2 = ConvBlock(16, 32)
        self.e3 = ConvBlock(32, 64)
        self.e4 = ConvBlock(64, 128)
        self.mid = ConvBlock(128, 256)
        self.u4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d4 = ConvBlock(256, 128)
        self.u3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d3 = ConvBlock(128, 64)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = ConvBlock(64, 32)
        self.u1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))
        x = self.d4(torch.cat([self.u4(z), d], dim=1))
        x = self.d3(torch.cat([self.u3(x), c], dim=1))
        x = self.d2(torch.cat([self.u2(x), b], dim=1))
        x = self.d1(torch.cat([self.u1(x), a], dim=1))
        return self.out(x)


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    model = SmallUNet().to(DEVICE)
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def preprocess(image):
    image = image.convert("RGB")
    w, h = image.size
    scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))

    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    left, top = (IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2
    canvas.paste(resized, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    return tensor, (left, top, nw, nh)


def predict(image, model):
    tensor, (left, top, nw, nh) = preprocess(image)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor.to(DEVICE)))[0, 0].cpu().numpy()

    probabilities = probabilities[top:top + nh, left:left + nw]
    p_img = Image.fromarray(
        np.clip(probabilities * 255, 0, 255).astype(np.uint8)
    ).resize(image.size, Image.Resampling.BILINEAR)

    probabilities = np.asarray(p_img, dtype=np.float32) / 255.0
    return probabilities >= MASK_THRESHOLD, probabilities


# ============================================================
# ROBUST POLYGON EXTRACTION
# ============================================================

def extract_polygon_points(canvas_json):
    """Return the last valid polygon's canvas coordinates, or None."""
    if not isinstance(canvas_json, dict):
        return None

    objects = canvas_json.get("objects", [])
    if not isinstance(objects, list):
        return None

    for obj in reversed(objects):
        if not isinstance(obj, dict):
            continue

        obj_type = str(obj.get("type", "")).lower()
        points = obj.get("points")

        # Accept polygons regardless of capitalization. Some canvas
        # versions may expose points even when type is missing.
        if not isinstance(points, list) or len(points) < 3:
            continue
        if obj_type and "polygon" not in obj_type:
            continue

        offset = obj.get("pathOffset") or {}
        ox = float(offset.get("x", 0))
        oy = float(offset.get("y", 0))
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        sx = float(obj.get("scaleX", 1))
        sy = float(obj.get("scaleY", 1))

        vertices = []
        for p in points:
            if not isinstance(p, dict) or "x" not in p or "y" not in p:
                continue

            # Fabric polygon points are local to pathOffset.
            x = (float(p["x"]) - ox) * sx + left
            y = (float(p["y"]) - oy) * sy + top
            vertices.append([x, y])

        if len(vertices) >= 3:
            return np.asarray(vertices, dtype=np.float32)

    return None


def polygon_to_roi(vertices, image_size, preview_size):
    image_w, image_h = image_size
    preview_w, preview_h = preview_size

    sx = image_w / preview_w
    sy = image_h / preview_h

    points = vertices.copy()
    points[:, 0] = np.clip(points[:, 0] * sx, 0, image_w - 1)
    points[:, 1] = np.clip(points[:, 1] * sy, 0, image_h - 1)

    roi = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(roi, [np.round(points).astype(np.int32)], 1)

    if int(roi.sum()) < 100:
        return None

    return roi.astype(bool)


def draw_belt_roi(image, key):
    image = image.convert("RGB")

    scale = min(850 / image.width, 850 / image.height, 1.0)
    preview_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    preview = image.resize(preview_size)

    st.caption(
        "Draw the polygon around the rubber belt only. "
        "Finish the polygon, then click Analyze below."
    )

    canvas = st_canvas(
        background_image=preview,
        height=preview.height,
        width=preview.width,
        drawing_mode="polygon",
        stroke_width=3,
        stroke_color="#00FF00",
        fill_color="rgba(0,255,0,0.20)",
        update_streamlit=True,
        key=key,
    )

    # Preserve canvas output between Streamlit reruns where possible.
    json_data = canvas.json_data
    if json_data:
        st.session_state[key + "_json"] = json_data
    else:
        json_data = st.session_state.get(key + "_json")

    vertices = extract_polygon_points(json_data)

    if vertices is None:
        st.info("No readable polygon was returned by the canvas yet.")
        return None

    roi = polygon_to_roi(vertices, image.size, preview_size)

    if roi is None:
        st.error("Polygon detected, but its area is too small. Draw it again.")
        return None

    st.success(
        f"Belt region detected: {int(roi.sum()):,} pixels "
        f"({roi.mean() * 100:.1f}% of the image)."
    )
    return roi


def overlay_mask(image, mask):
    arr = np.asarray(image.convert("RGB")).copy()
    arr[mask] = (
        0.60 * arr[mask] + 0.40 * np.array([255, 0, 0])
    ).astype(np.uint8)
    return Image.fromarray(arr)


def metrics_for(mask, roi, probabilities):
    belt_pixels = int(roi.sum())
    predicted_pixels = int(mask.sum())
    coverage = 100 * predicted_pixels / belt_pixels if belt_pixels else 0.0

    return {
        "belt_pixels": belt_pixels,
        "predicted_pixels": predicted_pixels,
        "predicted_coverage_percent": round(coverage, 5),
        "threshold": MASK_THRESHOLD,
    }


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# SEQUENTIAL DASHBOARD
# ============================================================

def frame_number(value):
    numbers = re.findall(r"\d+", Path(str(value)).stem)
    return int(numbers[-1]) if numbers else np.nan


@st.cache_data
def load_sequence_data():
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(f"Predictions not found: {PREDICTION_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    pred = pd.read_csv(PREDICTION_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    required = [
        "sequence_key", "current_damage",
        "predicted_future_crack_area_ratio",
        "future_crack_area_ratio_t1", "absolute_error",
        "predicted_damage_status", "predicted_damage_level",
    ]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    if "frame_key" not in pred.columns:
        if "frame" in pred.columns:
            pred["frame_key"] = pred["frame"]
        elif "canonical_image_filename" in pred.columns:
            pred["frame_key"] = pred["canonical_image_filename"].apply(frame_number)
        else:
            raise ValueError("No frame column found in prediction CSV.")

    pred["sequence_key"] = pred["sequence_key"].astype(str)
    pred["frame_key"] = pd.to_numeric(pred["frame_key"], errors="coerce")
    pred = pred.dropna(subset=["frame_key"]).copy()
    pred["frame_key"] = pred["frame_key"].astype(int)

    if "sequence" in manifest.columns:
        manifest["sequence_key"] = manifest["sequence"].astype(str)
    elif "sequence_key" not in manifest.columns:
        raise ValueError("Manifest has no sequence or sequence_key column.")

    manifest["frame_key"] = manifest["canonical_image_filename"].apply(frame_number)
    manifest["frame_key"] = pd.to_numeric(manifest["frame_key"], errors="coerce")
    manifest = manifest.dropna(subset=["frame_key"]).copy()
    manifest["frame_key"] = manifest["frame_key"].astype(int)

    return pred, manifest


def resolve_file(row, kind):
    source_col, filename_col = (
        ("source_image", "original_image_filename")
        if kind == "image"
        else ("source_json", "original_json_filename")
    )

    source = str(row.get(source_col, ""))
    if source and source.lower() not in ("nan", "none"):
        p = Path(source)
        if p.is_file():
            return p
        p = DATASET_ROOT / source
        if p.is_file():
            return p

    filename = str(row.get(filename_col, ""))
    if not filename or filename.lower() == "nan":
        return None

    for folder, _, files in os.walk(DATASET_ROOT):
        if filename in files:
            return Path(folder) / filename

    return None


def render_sequence():
    st.subheader("Sequential conveyor-belt analysis")
    pred, manifest = load_sequence_data()
    sequences = sorted(pred["sequence_key"].unique().tolist())

    if not sequences:
        st.warning("No sequences found.")
        return

    default = sequences.index("test/data24") if "test/data24" in sequences else 0
    sequence = st.selectbox("Select sequence", sequences, index=default)
    seq = pred[pred["sequence_key"] == sequence].sort_values("frame_key")
    frames = seq["frame_key"].astype(int).tolist()

    if not frames:
        st.warning("No frames found.")
        return

    frame = st.selectbox("Select frame", frames, index=len(frames) // 2)
    row = seq[seq["frame_key"] == frame].iloc[0]

    cols = st.columns(5)
    cols[0].metric("Current damage", f"{float(row['current_damage']):.6f}")
    cols[1].metric("Predicted next", f"{float(row['predicted_future_crack_area_ratio']):.6f}")
    cols[2].metric("Actual next", f"{float(row['future_crack_area_ratio_t1']):.6f}")
    cols[3].metric("Prediction error", f"{float(row['absolute_error']):.6f}")
    cols[4].metric("Frame", str(frame))

    c1, c2 = st.columns(2)
    c1.metric("Predicted status", str(row["predicted_damage_status"]))
    c2.metric("Predicted severity", str(row["predicted_damage_level"]))

    st.markdown("### Source image")
    matched = manifest[
        (manifest["sequence_key"] == sequence) &
        (manifest["frame_key"] == frame)
    ]

    if matched.empty:
        st.warning("Frame not found in the manifest.")
    else:
        mr = matched.iloc[0]
        image_path = resolve_file(mr, "image")
        json_path = resolve_file(mr, "json")

        if image_path is None:
            st.warning("Could not locate the source image.")
        else:
            image = Image.open(image_path).convert("RGB")
            st.image(image, use_container_width=True)

            if json_path is not None:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(image)
                    for shape in data.get("shapes", []):
                        if str(shape.get("label", "")).lower() == "crack":
                            pts = shape.get("points", [])
                            if len(pts) >= 3:
                                ax.add_patch(Polygon(pts, closed=True, fill=False))
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.warning(f"Could not display annotations: {exc}")

    st.markdown("### Progression")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(seq["frame_key"], seq["current_damage"], label="Current")
    ax.plot(seq["frame_key"], seq["predicted_future_crack_area_ratio"], label="Predicted next")
    ax.plot(seq["frame_key"], seq["future_crack_area_ratio_t1"], label="Actual next")
    ax.axvline(frame, linestyle="--")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Crack-area ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(seq, use_container_width=True, hide_index=True)
    st.download_button(
        "Download sequence CSV",
        seq.to_csv(index=False).encode("utf-8"),
        "sequence_damage_analysis.csv",
        "text/csv",
    )


# ============================================================
# GROQ
# ============================================================

def ask_groq(question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY is missing in the project-root .env file."

    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conveyor-belt inspection assistant. Answer using "
                "the supplied evidence. Model predictions are not verified "
                "physical damage. An empty mask does not prove safety."
            ),
        },
        {
            "role": "user",
            "content": "Current analysis:\n" + json.dumps(
                st.session_state.get("analysis_context", {}),
                default=str,
            ),
        },
    ]
    messages.extend(st.session_state.chat_history[-8:])
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=0.2,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Groq request failed: {exc}"


if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = {"status": "No media analyzed yet."}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# FOUR TABS
# ============================================================

tab_seq, tab_img, tab_vid, tab_ai = st.tabs([
    "Sequential dashboard",
    "Analyze image",
    "Analyze video",
    "Groq assistant",
])


# TAB 1: SEQUENTIAL
with tab_seq:
    try:
        render_sequence()
    except Exception as exc:
        st.error("Sequential dashboard error.")
        st.exception(exc)


# TAB 2: IMAGE
with tab_img:
    st.subheader("Image analysis")
    uploaded_image = st.file_uploader(
        "Upload conveyor-belt image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        roi = draw_belt_roi(image, "image_roi")

        # Button is always displayed, regardless of polygon detection.
        clicked = st.button("Analyze belt-only damage", type="primary", key="analyze_image")

        if clicked:
            if roi is None:
                st.error(
                    "The canvas did not provide a valid polygon. "
                    "Clear the drawing, draw one closed polygon, and try again."
                )
                with st.expander("Canvas diagnostic"):
                    st.json(st.session_state.get("image_roi_json", {}))
            else:
                try:
                    with st.spinner("Loading model and analyzing belt..."):
                        model = load_model()
                        raw, probs = predict(image, model)
                        final = raw & roi
                        metrics = metrics_for(final, roi, probs)

                    st.write("Raw model pixels:", int(raw.sum()))
                    st.write("Pixels inside belt polygon:", int(final.sum()))

                    mask_image = Image.fromarray(final.astype(np.uint8) * 255)
                    overlay = overlay_mask(image, final)

                    c1, c2 = st.columns(2)
                    c1.image(mask_image, caption="Predicted mask")
                    c2.image(overlay, caption="Red = predictions inside the belt")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted pixels", f"{metrics['predicted_pixels']:,}")
                    c2.metric("Belt pixels", f"{metrics['belt_pixels']:,}")
                    c3.metric("Predicted coverage", f"{metrics['predicted_coverage_percent']:.4f}%")

                    st.download_button(
                        "Download belt-only mask",
                        png_bytes(mask_image),
                        "belt_only_mask.png",
                        "image/png",
                    )

                    st.session_state.analysis_context = {
                        "media_type": "image",
                        "dimensions": list(image.size),
                        "metrics": metrics,
                        "threshold": MASK_THRESHOLD,
                        "warning": "Predictions are not verified physical damage.",
                    }
                    st.success("Analysis completed. Open Groq assistant to ask questions.")
                except Exception as exc:
                    st.error("Image analysis failed.")
                    st.exception(exc)


# TAB 3: VIDEO
with tab_vid:
    st.subheader("Video analysis")
    st.info("This workflow assumes the camera and belt position remain fixed.")
    uploaded_video = st.file_uploader(
        "Upload conveyor-belt video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video:
        st.video(uploaded_video)
        temp_path = None

        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.getvalue())
                temp_path = tmp.name

            cap = cv2.VideoCapture(temp_path)
            ok, first = cap.read()
            cap.release()

            if not ok:
                st.error("Could not read the first video frame.")
            else:
                first_image = Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
                st.image(first_image, caption="First frame", use_container_width=True)
                roi = draw_belt_roi(first_image, "video_roi")

                run_video = st.button("Analyze sampled video frames", type="primary", key="analyze_video")

                if run_video:
                    if roi is None:
                        st.error("No valid belt polygon detected. Draw it again.")
                        with st.expander("Canvas diagnostic"):
                            st.json(st.session_state.get("video_roi_json", {}))
                    else:
                        model = load_model()
                        cap = cv2.VideoCapture(temp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if not np.isfinite(fps) or fps <= 0:
                            fps = 25.0
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = total / fps if total > 0 else 0
                        n = min(MAX_VIDEO_FRAMES, max(1, int(duration) + 1))
                        indices = np.linspace(0, max(0, total - 1), n, dtype=int)

                        results = []
                        previews = []
                        progress = st.progress(0.0)

                        for i, index in enumerate(indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                            ok, frame = cap.read()
                            if ok:
                                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                if frame_image.size == first_image.size:
                                    raw, probs = predict(frame_image, model)
                                    final = raw & roi
                                    metrics = metrics_for(final, roi, probs)
                                    results.append({
                                        "frame": int(index),
                                        "time_seconds": round(float(index) / fps, 3),
                                        **metrics,
                                    })
                                    if len(previews) < 6:
                                        previews.append((int(index), overlay_mask(frame_image, final)))
                            progress.progress((i + 1) / len(indices))

                        cap.release()

                        if not results:
                            st.error("No frames could be analyzed.")
                        else:
                            df = pd.DataFrame(results)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                            fig, ax = plt.subplots(figsize=(11, 4))
                            ax.plot(df["time_seconds"], df["predicted_coverage_percent"], marker="o")
                            ax.set_xlabel("Time (seconds)")
                            ax.set_ylabel("Predicted belt coverage (%)")
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)

                            for index, preview in previews:
                                st.image(preview, caption=f"Frame {index}", use_container_width=True)

                            st.download_button(
                                "Download video CSV",
                                df.to_csv(index=False).encode("utf-8"),
                                "belt_video_results.csv",
                                "text/csv",
                            )

                            st.session_state.analysis_context = {
                                "media_type": "video",
                                "sampled_frames": len(df),
                                "mean_predicted_coverage_percent": round(float(df["predicted_coverage_percent"].mean()), 5),
                                "max_predicted_coverage_percent": round(float(df["predicted_coverage_percent"].max()), 5),
                                "frame_results": results,
                                "warning": "Predicted pixels are not verified physical damage.",
                            }
                            st.success("Video analysis completed. Open Groq assistant.")
        except Exception as exc:
            st.error("Video analysis failed.")
            st.exception(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# TAB 4: GROQ ASSISTANT
with tab_ai:
    st.subheader("Groq AI assistant")
    st.caption("Ask questions about the latest image/video analysis or the saved sequential evidence.")

    with st.expander("Current analysis evidence", expanded=True):
        st.json(st.session_state.analysis_context)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the analysis...")
    if question:
        previous = st.session_state.chat_history[-8:]
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Groq is answering..."):
                answer = ask_groq(question)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
