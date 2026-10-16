
import os
import re
import json
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

import torch
import torch.nn as nn

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

PREDICTIONS = (
    ROOT / "audit_outputs" /
    "stage_3_27_final_inference_predictions.csv"
)
MANIFEST = ROOT / "canonical_dataset" / "canonical_manifest.csv"
CHECKPOINT = (
    ROOT / "models" / "segmentation" /
    "beltcrack9kd_cpu_pilot_best.pt"
)
TEST_SUMMARY = (
    ROOT / "audit_outputs" / "stage_4_7_final_test" /
    "stage_4_7_test_summary.json"
)

IMAGE_SIZE = 256
THRESHOLD = 0.10
MAX_VIDEO_FRAMES = 60
SAMPLE_SECONDS = 1.0
DEVICE = torch.device("cpu")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="BeltCrack9kd Damage Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("BeltCrack9kd Damage Intelligence")
st.caption(
    "Sequential analysis | Image/video segmentation | Groq explanations"
)

with st.sidebar:
    st.header("Settings")
    st.write("Segmentation threshold:", THRESHOLD)
    st.write("Inference device: CPU")
    st.caption("Research prototype - manual inspection is required.")


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_predictions():
    if PREDICTIONS.is_file():
        return pd.read_csv(PREDICTIONS)
    return None


@st.cache_data(show_spinner=False)
def load_manifest():
    if not MANIFEST.is_file():
        return None

    df = pd.read_csv(MANIFEST)

    def frame_number(value):
        stem = Path(str(value)).stem
        nums = re.findall(r"\d+", stem)
        return int(nums[-1]) if nums else np.nan

    if "canonical_image_filename" in df.columns:
        df["frame_key"] = df["canonical_image_filename"].apply(frame_number)

    if "sequence" in df.columns:
        df["sequence_key"] = df["sequence"].astype(str)

    return df


def resolve_path(row, path_col, filename_col):
    source = str(row.get(path_col, "")).strip()

    if source and source.lower() not in ("nan", "none"):
        path = Path(source)
        if path.is_file():
            return path

        path = DATASET / source
        if path.is_file():
            return path

    filename = str(row.get(filename_col, "")).strip()

    if filename:
        for folder, _, files in os.walk(DATASET):
            if filename in files:
                return Path(folder) / filename

    return None


def get_polygons(json_path):
    if not json_path:
        return []

    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    polygons = []

    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).lower().strip() != "crack":
            continue

        points = shape.get("points", [])

        if len(points) >= 3:
            polygons.append(np.asarray(points, dtype=float))

    return polygons


# ============================================================
# SMALL U-NET: SAME ARCHITECTURE AS STAGE 4.6B
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


@st.cache_resource(show_spinner="Loading segmentation model...")
def load_model():
    if not CHECKPOINT.is_file():
        return None, f"Checkpoint not found: {CHECKPOINT}"

    model = SmallUNet().to(DEVICE)

    try:
        try:
            checkpoint = torch.load(
                CHECKPOINT,
                map_location=DEVICE,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval()

        return model, None

    except Exception as exc:
        return None, f"Checkpoint loading failed: {exc}"


# ============================================================
# IMAGE PREPROCESSING AND PREDICTION
# ============================================================

def preprocess(image):
    image = image.convert("RGB")
    width, height = image.size

    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    nw = max(1, round(width * scale))
    nh = max(1, round(height * scale))

    resized = image.resize(
        (nw, nh),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    left = (IMAGE_SIZE - nw) // 2
    top = (IMAGE_SIZE - nh) // 2
    canvas.paste(resized, (left, top))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0)

    return tensor, image.size, (nw, nh, left, top)


@torch.inference_mode()
def predict(model, image):
    tensor, original_size, geometry = preprocess(image)

    logits = model(tensor.to(DEVICE))
    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary = probabilities >= THRESHOLD

    nw, nh, left, top = geometry

    content_prob = probabilities[top:top + nh, left:left + nw]
    content_mask = binary[top:top + nh, left:left + nw]

    mask_image = Image.fromarray(
        content_mask.astype(np.uint8) * 255
    )
    mask_image = mask_image.resize(
        original_size, Image.Resampling.NEAREST
    )
    mask = np.asarray(mask_image) > 0

    rgb = np.asarray(image.convert("RGB"))
    overlay = rgb.copy()

    overlay[mask] = (
        0.55 * overlay[mask] +
        0.45 * np.array([255, 30, 30])
    ).astype(np.uint8)

    return {
        "mask": mask,
        "overlay": overlay,
        "coverage": float(content_mask.mean()),
        "max_probability": (
            float(content_prob.max()) if content_prob.size else 0.0
        ),
        "pixels": int(mask.sum()),
        "width": original_size[0],
        "height": original_size[1],
    }


# ============================================================
# GROQ ASSISTANT
# ============================================================

def groq_explain(question, evidence):
    key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not key:
        return None, "GROQ_API_KEY is missing from the project-root .env file."

    if Groq is None:
        return None, "Install Groq: python -m pip install groq"

    client = Groq(api_key=key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful ML assistant. Explain only the supplied "
                    "evidence. Do not invent metrics. SmallUNet predicts image "
                    "pixels; the GradientBoostingRegressor predicts future "
                    "crack-area ratios from engineered features. Groq does not "
                    "perform segmentation. The segmentation model has limited "
                    "test performance. A positive mask is not proof of damage, "
                    "and an empty mask does not prove the belt is safe."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evidence:\n"
                    + json.dumps(evidence, ensure_ascii=True, default=str)[:16000]
                    + "\n\nQuestion:\n"
                    + question
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    return response.choices[0].message.content, None


# ============================================================
# TAB 1: EXISTING SEQUENTIAL DASHBOARD
# ============================================================

def sequence_dashboard():
    st.header("Sequential Damage Intelligence")

    df = load_predictions()
    manifest = load_manifest()

    if df is None:
        st.error(f"Prediction file not found: {PREDICTIONS}")
        return

    required = [
        "sequence_key",
        "current_damage",
        "predicted_future_crack_area_ratio",
        "future_crack_area_ratio_t1",
        "absolute_error",
        "predicted_damage_status",
        "predicted_damage_level",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Missing columns in prediction CSV: {missing}")
        return

    if "frame_key" not in df.columns:
        if "frame" in df.columns:
            df["frame_key"] = pd.to_numeric(df["frame"], errors="coerce")
        elif "canonical_image_filename" in df.columns:
            df["frame_key"] = df["canonical_image_filename"].apply(
                lambda x: int(re.findall(r"\d+", Path(str(x)).stem)[-1])
                if re.findall(r"\d+", Path(str(x)).stem) else np.nan
            )
        else:
            df["frame_key"] = np.arange(len(df))

    sequences = sorted(df["sequence_key"].astype(str).unique())

    if not sequences:
        st.warning("No sequences found.")
        return

    default = sequences.index("test/data24") if "test/data24" in sequences else 0

    sequence = st.selectbox(
        "Select sequence", sequences, index=default
    )

    seq = df[df["sequence_key"].astype(str) == sequence].copy()
    seq["frame_key"] = pd.to_numeric(seq["frame_key"], errors="coerce")
    seq = seq.dropna(subset=["frame_key"]).sort_values("frame_key")

    if seq.empty:
        st.warning("No valid frames found.")
        return

    frames = seq["frame_key"].astype(int).tolist()

    frame = st.selectbox(
        "Select frame", frames,
        index=len(frames) // 2,
        key="sequence_frame",
    )

    row = seq[seq["frame_key"].astype(int) == frame].iloc[0]

    current = float(row["current_damage"])
    predicted = float(row["predicted_future_crack_area_ratio"])
    actual = float(row["future_crack_area_ratio_t1"])
    error = float(row["absolute_error"])
    status = str(row["predicted_damage_status"])
    severity = str(row["predicted_damage_level"])

    cols = st.columns(5)
    cols[0].metric("Current ratio", f"{current:.6f}")
    cols[1].metric("Predicted next", f"{predicted:.6f}")
    cols[2].metric("Recorded actual next", f"{actual:.6f}")
    cols[3].metric("Absolute error", f"{error:.6f}")
    cols[4].metric("Frame", frame)

    a, b = st.columns(2)
    a.info(f"Predicted status: **{status}**")
    b.info(f"Predicted severity: **{severity}**")

    if manifest is not None and {
        "sequence_key", "frame_key"
    }.issubset(manifest.columns):

        match = manifest[
            (manifest["sequence_key"].astype(str) == sequence)
            & (
                pd.to_numeric(manifest["frame_key"], errors="coerce")
                == frame
            )
        ]

        if len(match) == 1:
            mrow = match.iloc[0]

            image_path = resolve_path(
                mrow, "source_image", "original_image_filename"
            )
            json_path = resolve_path(
                mrow, "source_json", "original_json_filename"
            )

            if image_path:
                image = Image.open(image_path).convert("RGB")
                polygons = get_polygons(json_path)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(image)

                for polygon in polygons:
                    ax.add_patch(Polygon(
                        polygon, closed=True, fill=False, linewidth=2
                    ))

                ax.set_title("Recorded belt image and dataset annotations")
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.caption(
                    f"Dataset annotation polygons: {len(polygons)}. "
                    "These are not live model predictions."
                )

    st.subheader("Sequential damage progression")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seq["frame_key"], seq["current_damage"], label="Current damage")
    ax.plot(
        seq["frame_key"],
        seq["predicted_future_crack_area_ratio"],
        label="Predicted next damage",
    )
    ax.plot(
        seq["frame_key"],
        seq["future_crack_area_ratio_t1"],
        label="Recorded next damage",
    )
    ax.axvline(frame, linestyle="--")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Crack area ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    y_pred = seq["predicted_future_crack_area_ratio"].to_numpy(dtype=float)
    y_true = seq["future_crack_area_ratio_t1"].to_numpy(dtype=float)

    cols = st.columns(3)
    cols[0].metric("Sequence MAE", f"{np.mean(np.abs(y_pred-y_true)):.6f}")
    cols[1].metric(
        "Sequence RMSE",
        f"{np.sqrt(np.mean((y_pred-y_true)**2)):.6f}",
    )
    cols[2].metric("Frames", len(seq))

    left, right = st.columns(2)

    with left:
        st.subheader("Predicted status distribution")
        st.bar_chart(seq["predicted_damage_status"].value_counts())

    with right:
        st.subheader("Predicted severity distribution")
        st.bar_chart(seq["predicted_damage_level"].value_counts())

    st.subheader("Frame-level results")
    columns = [
        "frame_key", "current_damage",
        "predicted_future_crack_area_ratio",
        "future_crack_area_ratio_t1", "absolute_error",
        "predicted_damage_status", "predicted_damage_level",
    ]

    st.dataframe(
        seq[[c for c in columns if c in seq.columns]],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download sequence CSV",
        data=seq.to_csv(index=False),
        file_name=sequence.replace("/", "_") + "_damage_analysis.csv",
        mime="text/csv",
    )

    st.session_state["evidence"] = {
        "source": "existing sequential predictions",
        "sequence": sequence,
        "frame": frame,
        "current_damage_ratio": current,
        "predicted_future_crack_area_ratio": predicted,
        "recorded_future_crack_area_ratio": actual,
        "absolute_error": error,
        "predicted_status": status,
        "predicted_severity": severity,
        "sequence_frames": len(seq),
    }


# ============================================================
# TAB 2: UPLOAD IMAGE OR VIDEO
# ============================================================

def upload_analysis():
    st.header("Upload and analyze an image or video")

    st.warning(
        "The saved segmentation model has limited measured performance. "
        "Always verify the results manually."
    )

    model, error = load_model()

    if model is None:
        st.error(error)
        return

    uploaded = st.file_uploader(
        "Choose belt image or video",
        type=[
            "png", "jpg", "jpeg", "bmp", "tif", "tiff",
            "mp4", "avi", "mov", "mkv",
        ],
        key="uploaded_belt_file",
    )

    if uploaded is None:
        st.info("Upload an image or video to begin.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    video_types = {".mp4", ".avi", ".mov", ".mkv"}

    # ---------------- IMAGE ----------------
    if suffix not in video_types:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"Unable to read image: {exc}")
            return

        st.image(image, caption="Uploaded image", use_container_width=True)

        if not st.button("Analyze image", type="primary"):
            return

        with st.spinner("Running segmentation inference..."):
            result = predict(model, image)

        cols = st.columns(3)
        cols[0].metric(
            "Predicted mask coverage",
            f"{100 * result['coverage']:.3f}%",
        )
        cols[1].metric(
            "Maximum pixel probability",
            f"{result['max_probability']:.3f}",
        )
        cols[2].metric("Predicted pixels", f"{result['pixels']:,}")

        if result["pixels"] > 0:
            st.warning(
                "The model predicted crack-like pixels. This is not proof "
                "of physical damage."
            )
        else:
            st.info(
                "No pixels crossed the threshold. This does not prove "
                "the belt is undamaged."
            )

        a, b = st.columns(2)

        with a:
            st.image(
                result["mask"].astype(np.uint8) * 255,
                caption="Predicted crack mask",
                use_container_width=True,
            )

        with b:
            st.image(
                result["overlay"],
                caption="Overlay: red pixels are model predictions",
                use_container_width=True,
            )

        buffer = io.BytesIO()
        Image.fromarray(result["mask"].astype(np.uint8) * 255).save(
            buffer, format="PNG"
        )

        st.download_button(
            "Download predicted mask PNG",
            data=buffer.getvalue(),
            file_name="predicted_crack_mask.png",
            mime="image/png",
        )

        st.session_state["evidence"] = {
            "source": "uploaded image",
            "filename": uploaded.name,
            "width": result["width"],
            "height": result["height"],
            "segmentation_threshold": THRESHOLD,
            "predicted_mask_coverage_ratio": result["coverage"],
            "maximum_pixel_probability": result["max_probability"],
            "predicted_pixels": result["pixels"],
        }

    # ---------------- VIDEO ----------------
    else:
        if cv2 is None:
            st.error(
                "Install OpenCV first: "
                "python -m pip install opencv-python"
            )
            return

        st.caption(
            f"Samples up to {MAX_VIDEO_FRAMES} frames at "
            f"{SAMPLE_SECONDS:g}-second intervals."
        )

        if not st.button("Analyze video", type="primary"):
            return

        results = []
        temp_path = None
        capture = None
        progress = st.progress(0)
        message = st.empty()

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as temp:
                temp.write(uploaded.getvalue())
                temp_path = temp.name

            capture = cv2.VideoCapture(temp_path)

            if not capture.isOpened():
                st.error("Unable to open video. Try an H.264 MP4 file.")
                return

            fps = capture.get(cv2.CAP_PROP_FPS)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if not fps or fps <= 0:
                fps = 1.0

            duration = total / fps if total > 0 else 0.0
            times = np.arange(
                0, max(duration, 0.01), SAMPLE_SECONDS
            )[:MAX_VIDEO_FRAMES]

            for index, seconds in enumerate(times):
                capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds * 1000))
                ok, frame = capture.read()

                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                result = predict(model, pil_frame)

                results.append({
                    "seconds": float(seconds),
                    "coverage": result["coverage"],
                    "max_probability": result["max_probability"],
                    "pixels": result["pixels"],
                    "image": pil_frame,
                    "mask": result["mask"],
                    "overlay": result["overlay"],
                })

                progress.progress((index + 1) / max(len(times), 1))
                message.write(
                    f"Processed sample {index + 1}/{len(times)}"
                )

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            return

        finally:
            if capture is not None:
                capture.release()

            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if not results:
            st.error("No readable video frames were found.")
            return

        table = pd.DataFrame([
            {
                "seconds": r["seconds"],
                "predicted_coverage_ratio": r["coverage"],
                "maximum_pixel_probability": r["max_probability"],
                "predicted_pixels": r["pixels"],
            }
            for r in results
        ])

        mean_coverage = float(table["predicted_coverage_ratio"].mean())
        max_coverage = float(table["predicted_coverage_ratio"].max())
        flagged = int((table["predicted_pixels"] > 0).sum())

        cols = st.columns(3)
        cols[0].metric("Frames sampled", len(results))
        cols[1].metric(
            "Mean predicted coverage", f"{100*mean_coverage:.3f}%"
        )
        cols[2].metric("Frames with predicted pixels", f"{flagged}/{len(results)}")

        st.subheader("Predicted coverage across sampled frames")
        st.line_chart(
            table.set_index("seconds")[["predicted_coverage_ratio"]]
        )

        index = st.select_slider(
            "Inspect video frame",
            options=list(range(len(results))),
            value=0,
            format_func=lambda i: (
                f"Sample {i+1} at {results[i]['seconds']:.1f} seconds"
            ),
        )

        selected = results[index]
        a, b, c = st.columns(3)

        with a:
            st.image(selected["image"], caption="Video frame",
                     use_container_width=True)
        with b:
            st.image(selected["mask"].astype(np.uint8) * 255,
                     caption="Predicted mask", use_container_width=True)
        with c:
            st.image(selected["overlay"], caption="Prediction overlay",
                     use_container_width=True)

        st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button(
            "Download video results CSV",
            data=table.to_csv(index=False),
            file_name="video_damage_analysis.csv",
            mime="text/csv",
        )

        st.session_state["evidence"] = {
            "source": "uploaded video",
            "filename": uploaded.name,
            "sampled_frames": len(results),
            "sample_interval_seconds": SAMPLE_SECONDS,
            "mean_predicted_coverage_ratio": mean_coverage,
            "maximum_predicted_coverage_ratio": max_coverage,
            "frames_with_predicted_pixels": flagged,
            "frame_results": table.to_dict(orient="records"),
        }


# ============================================================
# TAB 3: GROQ
# ============================================================

def groq_tab():
    st.header("Groq Damage Assistant")

    summary = load_json(TEST_SUMMARY)

    st.info(
        "Recorded held-out segmentation test metrics: mean image Dice "
        "0.160466, mean image precision 0.135608, mean image recall "
        "0.237736, global pixel Dice 0.241551, global pixel precision "
        "0.232126, global pixel recall 0.251773. These results indicate "
        "limited segmentation performance."
    )

    if summary:
        with st.expander("Full recorded test summary"):
            st.json(summary)

    evidence = st.session_state.get("evidence")

    if evidence:
        with st.expander("Current analysis evidence"):
            st.json(evidence)
    else:
        st.warning(
            "First analyze an upload or select a sequence in the first tab."
        )

    question = st.text_area(
        "Ask about the results",
        placeholder=(
            "Explain the predicted mask coverage and the limitations "
            "of using it to assess belt damage."
        ),
    )

    if st.button("Explain with Groq", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
            return

        if not evidence:
            st.warning("No current analysis evidence is available.")
            return

        combined = {
            "current_analysis": evidence,
            "recorded_segmentation_test_metrics": summary,
        }

        try:
            with st.spinner("Groq is preparing an explanation..."):
                answer, error = groq_explain(question, combined)

            if error:
                st.error(error)
            else:
                st.subheader("Explanation")
                st.markdown(answer)

        except Exception as exc:
            st.error(
                "Groq request failed. Check your API key, model setting, "
                "network connection, and account limits."
            )
            st.caption(f"Error type: {type(exc).__name__}")


# ============================================================
# MAIN APPLICATION
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Sequential dashboard",
    "Upload image / video",
    "Groq assistant",
])

with tab1:
    sequence_dashboard()

with tab2:
    upload_analysis()

with tab3:
    groq_tab()

st.divider()
st.caption(
    "Research prototype only. Model predictions are experimental and must "
    "not be used as the sole basis for maintenance or safety decisions."
)
