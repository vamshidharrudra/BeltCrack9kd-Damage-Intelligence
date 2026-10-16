import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ============================================================
# STAGE 4.1
# BELTCRACK9KD DAMAGE INTELLIGENCE DASHBOARD
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

DATASET_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_27_final_inference_predictions.csv"
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "beltcrack9kd_final_gbr_33_features.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BeltCrack9kd Damage Intelligence",
    page_icon="",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        text-align: center;
        background-color: #fafafa;
    }

    .metric-title {
        font-size: 14px;
        color: #666666;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
    }

    .status-box {
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        border: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_frame(value):

    value = str(value)

    stem = os.path.splitext(
        os.path.basename(value)
    )[0]

    if stem.isdigit():
        return int(stem)

    numbers = re.findall(
        r"\d+",
        stem
    )

    if not numbers:
        return np.nan

    return int(numbers[-1])


def resolve_image_path(row):

    candidates = []

    if "source_image" in row.index:

        value = str(
            row["source_image"]
        )

        if value not in [
            "",
            "nan",
            "None"
        ]:

            candidates.append(value)


    for candidate in candidates:

        if os.path.exists(candidate):

            return candidate


    filename = str(
        row[
            "original_image_filename"
        ]
    )


    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )


    return None


def resolve_json_path(row):

    if "source_json" in row.index:

        source_json = str(
            row["source_json"]
        )

        if os.path.exists(
            source_json
        ):

            return source_json


    filename = str(
        row[
            "original_json_filename"
        ]
    )


    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )


    return None


def load_polygons(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    polygons = []

    for shape in data.get(
        "shapes",
        []
    ):

        label = str(
            shape.get(
                "label",
                ""
            )
        ).lower().strip()


        if label != "crack":
            continue


        points = shape.get(
            "points",
            []
        )


        if len(points) >= 3:

            polygons.append(
                np.asarray(
                    points,
                    dtype=float
                )
            )


    return polygons


def make_belt_visualization(
    image_path,
    json_path,
    frame,
    current_damage,
    predicted_damage,
    actual_damage,
    error,
    status,
    severity
):

    image = plt.imread(
        image_path
    )

    polygons = load_polygons(
        json_path
    )


    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.imshow(
        image
    )


    for polygon in polygons:

        patch = Polygon(
            polygon,
            closed=True,
            fill=False,
            linewidth=2
        )

        ax.add_patch(
            patch
        )


    ax.set_title(
        "BeltCrack9kd Damage Intelligence",
        fontsize=17,
        fontweight="bold"
    )


    info = (
        f"Frame: {frame}\n"
        f"Cracks: {len(polygons)}\n"
        f"Current damage: {current_damage:.6f}\n"
        f"Predicted next: {predicted_damage:.6f}\n"
        f"Actual next: {actual_damage:.6f}\n"
        f"Error: {error:.6f}\n"
        f"Status: {status}\n"
        f"Severity: {severity}"
    )


    ax.text(
        0.02,
        0.03,
        info,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            alpha=0.88
        )
    )


    ax.axis(
        "off"
    )

    plt.tight_layout()

    return fig, len(polygons)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_predictions():

    return pd.read_csv(
        PREDICTION_PATH
    )


@st.cache_data
def load_manifest():

    df = pd.read_csv(
        MANIFEST_PATH
    )

    df["frame_key"] = df[
        "canonical_image_filename"
    ].apply(
        extract_frame
    )

    df["sequence_key"] = df[
        "sequence"
    ].astype(str)

    return df


@st.cache_resource
def load_model():

    artifact = joblib.load(
        MODEL_PATH
    )

    if isinstance(
        artifact,
        dict
    ):

        return artifact[
            "model"
        ]

    return artifact


predictions = load_predictions()

manifest = load_manifest()

model = load_model()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'BeltCrack9kd Damage Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sequential conveyor-belt crack analysis and next-frame damage prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Control Panel"
)


sequences = sorted(
    predictions[
        "sequence_key"
    ].astype(str).unique()
)


selected_sequence = st.sidebar.selectbox(
    "Select sequence",
    sequences,
    index=(
        sequences.index(
            "test/data24"
        )
        if "test/data24" in sequences
        else 0
    )
)


sequence_df = predictions[
    predictions[
        "sequence_key"
    ].astype(str)
    ==
    selected_sequence
].copy()


sequence_df = sequence_df.sort_values(
    "frame_key"
    if "frame_key" in sequence_df.columns
    else "frame"
).reset_index(
    drop=True
)


if "frame_key" not in sequence_df.columns:

    sequence_df["frame_key"] = (
        sequence_df[
            "frame"
        ]
        if "frame" in sequence_df.columns
        else sequence_df[
            "canonical_image_filename"
        ].apply(
            extract_frame
        )
    )


frames = sequence_df[
    "frame_key"
].astype(int).tolist()


selected_frame = st.sidebar.selectbox(
    "Select frame",
    frames,
    index=len(frames) // 2
)


# ============================================================
# SELECTED ROW
# ============================================================

row = sequence_df[
    sequence_df[
        "frame_key"
    ]
    ==
    selected_frame
].iloc[0]


current_damage = float(
    row[
        "current_damage"
    ]
)

predicted_damage = float(
    row[
        "predicted_future_crack_area_ratio"
    ]
)

actual_damage = float(
    row[
        "future_crack_area_ratio_t1"
    ]
)

error = float(
    row[
        "absolute_error"
    ]
)

status = str(
    row[
        "predicted_damage_status"
    ]
)

severity = str(
    row[
        "predicted_damage_level"
    ]
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    "Current Frame Intelligence"
)


col1, col2, col3, col4, col5 = st.columns(
    5
)


with col1:

    st.metric(
        "Current Damage",
        f"{current_damage:.6f}"
    )


with col2:

    st.metric(
        "Predicted Next",
        f"{predicted_damage:.6f}"
    )


with col3:

    st.metric(
        "Actual Next",
        f"{actual_damage:.6f}"
    )


with col4:

    st.metric(
        "Prediction Error",
        f"{error:.6f}"
    )


with col5:

    st.metric(
        "Frame",
        selected_frame
    )


# ============================================================
# STATUS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


col1, col2 = st.columns(
    2
)


with col1:

    st.markdown(
        f"""
        <div class="status-box">
        Predicted Status<br>
        {status}
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="status-box">
        Predicted Severity<br>
        {severity}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BELT IMAGE
# ============================================================

st.subheader(
    "Actual Conveyor Belt and Crack Region"
)


manifest_match = manifest[
    (
        manifest[
            "sequence_key"
        ]
        ==
        selected_sequence
    )
    &
    (
        manifest[
            "frame_key"
        ]
        ==
        selected_frame
    )
]


if len(manifest_match) == 1:

    manifest_row = (
        manifest_match.iloc[0]
    )

    image_path = resolve_image_path(
        manifest_row
    )

    json_path = resolve_json_path(
        manifest_row
    )


    if (
        image_path is not None
        and
        json_path is not None
    ):

        fig, crack_count = (
            make_belt_visualization(
                image_path,
                json_path,
                selected_frame,
                current_damage,
                predicted_damage,
                actual_damage,
                error,
                status,
                severity
            )
        )

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(
            fig
        )

        st.caption(
            f"Detected/annotated crack polygons: {crack_count}"
        )

    else:

        st.error(
            "Could not resolve belt image or annotation."
        )

else:

    st.error(
        "Frame not found in canonical manifest."
    )


# ============================================================
# DAMAGE PROGRESSION
# ============================================================

st.subheader(
    "Sequential Damage Progression"
)


fig, ax = plt.subplots(
    figsize=(13, 5)
)


ax.plot(
    sequence_df[
        "frame_key"
    ],
    sequence_df[
        "current_damage"
    ],
    label="Current damage",
    linewidth=2
)


ax.plot(
    sequence_df[
        "frame_key"
    ],
    sequence_df[
        "predicted_future_crack_area_ratio"
    ],
    label="Predicted next damage",
    linewidth=2
)


ax.plot(
    sequence_df[
        "frame_key"
    ],
    sequence_df[
        "future_crack_area_ratio_t1"
    ],
    label="Actual next damage",
    linewidth=2
)


ax.axvline(
    selected_frame,
    linestyle="--",
    linewidth=1.5
)


ax.set_xlabel(
    "Frame"
)

ax.set_ylabel(
    "Crack area ratio"
)

ax.set_title(
    "Damage Progression"
)

ax.grid(
    alpha=0.3
)

ax.legend()


st.pyplot(
    fig,
    width="stretch"
)

plt.close(
    fig
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "Sequence Performance"
)


sequence_mae = np.mean(
    np.abs(
        sequence_df[
            "predicted_future_crack_area_ratio"
        ]
        -
        sequence_df[
            "future_crack_area_ratio_t1"
        ]
    )
)


sequence_rmse = np.sqrt(
    np.mean(
        (
            sequence_df[
                "predicted_future_crack_area_ratio"
            ]
            -
            sequence_df[
                "future_crack_area_ratio_t1"
            ]
        ) ** 2
    )
)


c1, c2, c3 = st.columns(
    3
)


with c1:

    st.metric(
        "Sequence MAE",
        f"{sequence_mae:.6f}"
    )


with c2:

    st.metric(
        "Sequence RMSE",
        f"{sequence_rmse:.6f}"
    )


with c3:

    st.metric(
        "Frames",
        len(sequence_df)
    )


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

st.subheader(
    "Predicted Status Distribution"
)


status_counts = (
    sequence_df[
        "predicted_damage_status"
    ]
    .value_counts()
)


st.bar_chart(
    status_counts
)


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

st.subheader(
    "Predicted Severity Distribution"
)


severity_counts = (
    sequence_df[
        "predicted_damage_level"
    ]
    .value_counts()
)


st.bar_chart(
    severity_counts
)


# ============================================================
# FRAME DATA
# ============================================================

st.subheader(
    "Frame-Level Intelligence"
)


display_columns = [

    "frame_key",

    "current_damage",

    "predicted_future_crack_area_ratio",

    "future_crack_area_ratio_t1",

    "absolute_error",

    "predicted_damage_status",

    "predicted_damage_level"
]


display_columns = [
    c
    for c in display_columns
    if c in sequence_df.columns
]


st.dataframe(
    sequence_df[
        display_columns
    ],
    width="stretch",
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = sequence_df.to_csv(
    index=False
)


st.download_button(
    "Download Sequence Analysis CSV",
    data=csv_data,
    file_name=(
        selected_sequence
        .replace(
            "/",
            "_"
        )
        +
        "_damage_analysis.csv"
    ),
    mime="text/csv"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "Model: GradientBoostingRegressor"
    )

    st.write(
        "Final feature count: 33"
    )

    st.write(
        "Target: future crack area ratio at t+1"
    )

    st.write(
        "Model source: Stage 3.26 final artifact"
    )

    st.write(
        "Inference source: Stage 3.27 final inference"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application reports an annotation-derived "
    "crack-area damage proxy. Spatial crack continuity "
    "is based on heuristic polygon matching and should "
    "not be interpreted as a certified physical inspection "
    "or maintenance decision system."
)