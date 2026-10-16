import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 2.1
# BASELINE ERROR AND RESIDUAL ANALYSIS
#
# Purpose:
#   Analyze Stage 2.0 predictions without retraining models.
#
# Questions:
#   1. Which model performs best per sequence?
#   2. Which model performs best per component?
#   3. When does ML beat persistence?
#   4. When does persistence beat ML?
#   5. Are errors biased toward high/low damage?
#   6. Does the model overpredict or underpredict?
#   7. Which sequences are difficult?
#   8. How much does each model improve/worsen over persistence?
#
# No model training is performed.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_0_baseline_predictions.csv"
)

RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_0_baseline_results.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_0_baseline_ml_summary.json"
)

OUTPUT_SEQUENCE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_sequence_error_analysis.csv"
)

OUTPUT_COMPONENT = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_component_error_analysis.csv"
)

OUTPUT_DAMAGE_BIN = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_damage_level_error_analysis.csv"
)

OUTPUT_COMPARISON = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_model_comparison.csv"
)

OUTPUT_CASES = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_prediction_cases.csv"
)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_error_analysis_summary.json"
)


# ============================================================
# START
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.1 - BASELINE ERROR AND RESIDUAL ANALYSIS")
print("=" * 70)
print("")


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    (
        PREDICTION_FILE,
        "Stage 2.0 prediction file"
    ),
    (
        RESULT_FILE,
        "Stage 2.0 result file"
    ),
    (
        SUMMARY_FILE,
        "Stage 2.0 summary file"
    )
]


for path, description in required_files:

    if not os.path.exists(path):

        print(
            "ERROR:",
            description,
            "not found:"
        )

        print(path)

        raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading Stage 2.0 predictions..."
)

predictions = pd.read_csv(
    PREDICTION_FILE
)

print(
    "Prediction records:",
    len(predictions)
)


print("")
print(
    "Loading Stage 2.0 results..."
)

results = pd.read_csv(
    RESULT_FILE
)

print(
    "Result records:",
    len(results)
)


with open(
    SUMMARY_FILE,
    "r",
    encoding="utf-8"
) as file:

    stage_2_summary = json.load(
        file
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "model",
    "record_id",
    "sequence",
    "component",
    "current_frame",
    "future_frame",
    "current_crack_area_ratio",
    "actual_target_crack_area_ratio",
    "predicted_target_crack_area_ratio"
]


missing_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]


if missing_columns:

    print("")
    print(
        "ERROR: Missing prediction columns:"
    )

    for column in missing_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# BASIC INFORMATION
# ============================================================

models = sorted(
    predictions[
        "model"
    ]
    .dropna()
    .unique()
    .tolist()
)


print("")
print(
    "Models found:"
)

for model in models:

    print(
        "  -",
        model
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "record_id",
    "component",
    "current_frame",
    "future_frame",
    "current_crack_area_ratio",
    "actual_target_crack_area_ratio",
    "predicted_target_crack_area_ratio"
]


for column in numeric_columns:

    predictions[
        column
    ] = pd.to_numeric(
        predictions[
            column
        ],
        errors="coerce"
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

print("")
print(
    "Checking prediction values..."
)


missing_prediction_values = int(
    predictions[
        numeric_columns
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "Missing numeric values:",
    missing_prediction_values
)


if missing_prediction_values != 0:

    print(
        "ERROR: Missing prediction values."
    )

    raise SystemExit(1)


# ============================================================
# CALCULATE RESIDUALS
#
# residual = prediction - actual
#
# positive residual = overprediction
# negative residual = underprediction
# ============================================================

predictions[
    "residual"
] = (
    predictions[
        "predicted_target_crack_area_ratio"
    ]
    -
    predictions[
        "actual_target_crack_area_ratio"
    ]
)


predictions[
    "absolute_error"
] = (
    predictions[
        "residual"
    ]
    .abs()
)


predictions[
    "squared_error"
] = (
    predictions[
        "residual"
    ]
    ** 2
)


predictions[
    "current_to_future_change"
] = (
    predictions[
        "actual_target_crack_area_ratio"
    ]
    -
    predictions[
        "current_crack_area_ratio"
    ]
)


# ============================================================
# CHECK FRAME DIRECTION
# ============================================================

invalid_frame_direction = int(
    (
        predictions[
            "future_frame"
        ]
        <=
        predictions[
            "current_frame"
        ]
    ).sum()
)


print(
    "Invalid future frame direction:",
    invalid_frame_direction
)


if invalid_frame_direction != 0:

    print(
        "ERROR: Invalid temporal direction."
    )

    raise SystemExit(1)


# ============================================================
# MODEL-LEVEL ERROR ANALYSIS
# ============================================================

print("")
print("=" * 70)
print("MODEL-LEVEL ERROR ANALYSIS")
print("=" * 70)


model_summary_records = []


for model in models:

    subset = predictions[
        predictions[
            "model"
        ]
        ==
        model
    ].copy()


    actual = subset[
        "actual_target_crack_area_ratio"
    ]

    predicted = subset[
        "predicted_target_crack_area_ratio"
    ]


    mae = float(
        subset[
            "absolute_error"
        ].mean()
    )


    rmse = float(
        np.sqrt(
            subset[
                "squared_error"
            ].mean()
        )
    )


    residual_mean = float(
        subset[
            "residual"
        ].mean()
    )


    residual_median = float(
        subset[
            "residual"
        ].median()
    )


    overprediction_fraction = float(
        (
            subset[
                "residual"
            ]
            >
            0
        ).mean()
    )


    underprediction_fraction = float(
        (
            subset[
                "residual"
            ]
            <
            0
        ).mean()
    )


    exact_fraction = float(
        (
            subset[
                "residual"
            ]
            ==
            0
        ).mean()
    )


    max_absolute_error = float(
        subset[
            "absolute_error"
        ].max()
    )


    q95_absolute_error = float(
        subset[
            "absolute_error"
        ].quantile(
            0.95
        )
    )


    model_summary_records.append({

        "model":
            model,

        "records":
            int(
                len(
                    subset
                )
            ),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "mean_residual":
            residual_mean,

        "median_residual":
            residual_median,

        "overprediction_fraction":
            overprediction_fraction,

        "underprediction_fraction":
            underprediction_fraction,

        "zero_error_fraction":
            exact_fraction,

        "max_absolute_error":
            max_absolute_error,

        "p95_absolute_error":
            q95_absolute_error
    })


model_summary = pd.DataFrame(
    model_summary_records
)


print(
    model_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# SEQUENCE-LEVEL ANALYSIS
# ============================================================

print("")
print("=" * 70)
print("SEQUENCE-LEVEL ERROR ANALYSIS")
print("=" * 70)


sequence_records = []


for model in models:

    model_data = predictions[
        predictions[
            "model"
        ]
        ==
        model
    ]


    for sequence in sorted(
        model_data[
            "sequence"
        ].unique()
    ):

        subset = model_data[
            model_data[
                "sequence"
            ]
            ==
            sequence
        ]


        sequence_records.append({

            "model":
                model,

            "sequence":
                sequence,

            "records":
                int(
                    len(
                        subset
                    )
                ),

            "MAE":
                float(
                    subset[
                        "absolute_error"
                    ].mean()
                ),

            "RMSE":
                float(
                    np.sqrt(
                        subset[
                            "squared_error"
                        ].mean()
                    )
                ),

            "mean_residual":
                float(
                    subset[
                        "residual"
                    ].mean()
                ),

            "median_absolute_error":
                float(
                    subset[
                        "absolute_error"
                    ].median()
                ),

            "max_absolute_error":
                float(
                    subset[
                        "absolute_error"
                    ].max()
                ),

            "overprediction_fraction":
                float(
                    (
                        subset[
                            "residual"
                        ]
                        >
                        0
                    ).mean()
                ),

            "underprediction_fraction":
                float(
                    (
                        subset[
                            "residual"
                        ]
                        <
                        0
                    ).mean()
                )
        })


sequence_df = pd.DataFrame(
    sequence_records
)


sequence_df.to_csv(
    OUTPUT_SEQUENCE,
    index=False
)


# ============================================================
# COMPONENT-LEVEL ANALYSIS
# ============================================================

print("")
print("=" * 70)
print("COMPONENT-LEVEL ERROR ANALYSIS")
print("=" * 70)


component_records = []


for model in models:

    model_data = predictions[
        predictions[
            "model"
        ]
        ==
        model
    ]


    for component in sorted(
        model_data[
            "component"
        ].unique()
    ):

        subset = model_data[
            model_data[
                "component"
            ]
            ==
            component
        ]


        component_records.append({

            "model":
                model,

            "component":
                int(
                    component
                ),

            "records":
                int(
                    len(
                        subset
                    )
                ),

            "MAE":
                float(
                    subset[
                        "absolute_error"
                    ].mean()
                ),

            "RMSE":
                float(
                    np.sqrt(
                        subset[
                            "squared_error"
                        ].mean()
                    )
                ),

            "mean_residual":
                float(
                    subset[
                        "residual"
                    ].mean()
                ),

            "max_absolute_error":
                float(
                    subset[
                        "absolute_error"
                    ].max()
                )
        })


component_df = pd.DataFrame(
    component_records
)


component_df.to_csv(
    OUTPUT_COMPONENT,
    index=False
)


# ============================================================
# DAMAGE LEVEL ANALYSIS
#
# Bins are NOT severity labels.
#
# They are only analysis groups based on the CURRENT
# crack_area_ratio.
#
# This does not change the prediction target.
# ============================================================

print("")
print("=" * 70)
print("CURRENT DAMAGE LEVEL ERROR ANALYSIS")
print("=" * 70)


damage_bins = [
    -np.inf,
    0.0025,
    0.0050,
    0.0100,
    np.inf
]


damage_labels = [
    "very_low",
    "low",
    "moderate",
    "high"
]


predictions[
    "current_damage_level"
] = pd.cut(
    predictions[
        "current_crack_area_ratio"
    ],
    bins=damage_bins,
    labels=damage_labels,
    include_lowest=True
)


damage_records = []


for model in models:

    model_data = predictions[
        predictions[
            "model"
        ]
        ==
        model
    ]


    for damage_level in damage_labels:

        subset = model_data[
            model_data[
                "current_damage_level"
            ]
            ==
            damage_level
        ]


        if len(
            subset
        ) == 0:

            continue


        damage_records.append({

            "model":
                model,

            "current_damage_level":
                damage_level,

            "records":
                int(
                    len(
                        subset
                    )
                ),

            "current_mean":
                float(
                    subset[
                        "current_crack_area_ratio"
                    ].mean()
                ),

            "target_mean":
                float(
                    subset[
                        "actual_target_crack_area_ratio"
                    ].mean()
                ),

            "MAE":
                float(
                    subset[
                        "absolute_error"
                    ].mean()
                ),

            "RMSE":
                float(
                    np.sqrt(
                        subset[
                            "squared_error"
                        ].mean()
                    )
                ),

            "mean_residual":
                float(
                    subset[
                        "residual"
                    ].mean()
                ),

            "overprediction_fraction":
                float(
                    (
                        subset[
                            "residual"
                        ]
                        >
                        0
                    ).mean()
                ),

            "underprediction_fraction":
                float(
                    (
                        subset[
                            "residual"
                        ]
                        <
                        0
                    ).mean()
                )
        })


damage_df = pd.DataFrame(
    damage_records
)


damage_df.to_csv(
    OUTPUT_DAMAGE_BIN,
    index=False
)


# ============================================================
# ML VS PERSISTENCE
#
# Compare each ML prediction with the persistence prediction
# for the exact same test records.
# ============================================================

print("")
print("=" * 70)
print("MODEL VS PERSISTENCE ANALYSIS")
print("=" * 70)


test_predictions = predictions.copy()


test_predictions[
    "is_test"
] = True


persistence_data = test_predictions[
    test_predictions[
        "model"
    ]
    ==
    "Persistence"
].copy()


persistence_data = persistence_data[
    [
        "record_id",
        "actual_target_crack_area_ratio",
        "predicted_target_crack_area_ratio"
    ]
].copy()


persistence_data = (
    persistence_data
    .rename(
        columns={
            "predicted_target_crack_area_ratio":
                "persistence_prediction"
        }
    )
)


ml_data = test_predictions[
    test_predictions[
        "model"
    ]
    !=
    "Persistence"
].copy()


comparison = ml_data.merge(
    persistence_data[
        [
            "record_id",
            "persistence_prediction"
        ]
    ],
    on="record_id",
    how="inner"
)


comparison[
    "persistence_absolute_error"
] = (
    comparison[
        "persistence_prediction"
    ]
    -
    comparison[
        "actual_target_crack_area_ratio"
    ]
).abs()


comparison[
    "ml_absolute_error"
] = (
    comparison[
        "predicted_target_crack_area_ratio"
    ]
    -
    comparison[
        "actual_target_crack_area_ratio"
    ]
).abs()


comparison[
    "absolute_error_difference"
] = (
    comparison[
        "ml_absolute_error"
    ]
    -
    comparison[
        "persistence_absolute_error"
    ]
)


comparison[
    "ml_beats_persistence"
] = (
    comparison[
        "ml_absolute_error"
    ]
    <
    comparison[
        "persistence_absolute_error"
    ]
)


comparison[
    "persistence_beats_ml"
] = (
    comparison[
        "persistence_absolute_error"
    ]
    <
    comparison[
        "ml_absolute_error"
    ]
)


comparison[
    "tie"
] = (
    comparison[
        "ml_absolute_error"
    ]
    ==
    comparison[
        "persistence_absolute_error"
    ]
)


comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ============================================================
# MODEL-BY-MODEL PERSISTENCE COMPARISON
# ============================================================

comparison_summary_records = []


for model in sorted(
    comparison[
        "model"
    ].unique()
):

    subset = comparison[
        comparison[
            "model"
        ]
        ==
        model
    ]


    comparison_summary_records.append({

        "model":
            model,

        "records":
            int(
                len(
                    subset
                )
            ),

        "ml_mean_absolute_error":
            float(
                subset[
                    "ml_absolute_error"
                ].mean()
            ),

        "persistence_mean_absolute_error":
            float(
                subset[
                    "persistence_absolute_error"
                ].mean()
            ),

        "ml_beats_persistence_fraction":
            float(
                subset[
                    "ml_beats_persistence"
                ].mean()
            ),

        "persistence_beats_ml_fraction":
            float(
                subset[
                    "persistence_beats_ml"
                ].mean()
            ),

        "tie_fraction":
            float(
                subset[
                    "tie"
                ].mean()
            ),

        "mean_absolute_error_difference":
            float(
                subset[
                    "absolute_error_difference"
                ].mean()
            )
    })


comparison_summary = pd.DataFrame(
    comparison_summary_records
)


print(
    comparison_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# FIND BEST/WORST SEQUENCES FOR EACH MODEL
# ============================================================

sequence_ranking_records = []


for model in models:

    model_sequence = sequence_df[
        sequence_df[
            "model"
        ]
        ==
        model
    ].copy()


    if len(
        model_sequence
    ) == 0:

        continue


    best_row = model_sequence.loc[
        model_sequence[
            "MAE"
        ].idxmin()
    ]


    worst_row = model_sequence.loc[
        model_sequence[
            "MAE"
        ].idxmax()
    ]


    sequence_ranking_records.append({

        "model":
            model,

        "best_sequence":
            best_row[
                "sequence"
            ],

        "best_sequence_MAE":
            float(
                best_row[
                    "MAE"
                ]
            ),

        "worst_sequence":
            worst_row[
                "sequence"
            ],

        "worst_sequence_MAE":
            float(
                worst_row[
                    "MAE"
                ]
            )
    })


sequence_ranking_df = pd.DataFrame(
    sequence_ranking_records
)


# ============================================================
# MOST DIFFICULT TEST CASES
# ============================================================

print("")
print("=" * 70)
print("MOST DIFFICULT TEST CASES")
print("=" * 70)


test_only = predictions[
    predictions[
        "model"
    ]
    !=
    "Persistence"
].copy()


# For each ML model, retain its largest errors.

difficult_cases = []


for model in sorted(
    test_only[
        "model"
    ].unique()
):

    model_test = test_only[
        test_only[
            "model"
        ]
        ==
        model
    ].copy()


    model_test = model_test.sort_values(
        by="absolute_error",
        ascending=False
    )


    top_cases = model_test.head(
        20
    )


    for _, row in top_cases.iterrows():

        difficult_cases.append({

            "model":
                model,

            "record_id":
                int(
                    row[
                        "record_id"
                    ]
                ),

            "sequence":
                row[
                    "sequence"
                ],

            "component":
                int(
                    row[
                        "component"
                    ]
                ),

            "current_frame":
                int(
                    row[
                        "current_frame"
                    ]
                ),

            "future_frame":
                int(
                    row[
                        "future_frame"
                    ]
                ),

            "current_damage":
                float(
                    row[
                        "current_crack_area_ratio"
                    ]
                ),

            "actual_future_damage":
                float(
                    row[
                        "actual_target_crack_area_ratio"
                    ]
                ),

            "predicted_future_damage":
                float(
                    row[
                        "predicted_target_crack_area_ratio"
                    ]
                ),

            "absolute_error":
                float(
                    row[
                        "absolute_error"
                    ]
                ),

            "residual":
                float(
                    row[
                        "residual"
                    ]
                )
        })


difficult_cases_df = pd.DataFrame(
    difficult_cases
)


difficult_cases_df.to_csv(
    OUTPUT_CASES,
    index=False
)


# ============================================================
# DAMAGE CHANGE ANALYSIS
#
# This tells us whether the models struggle particularly when
# damage changes sharply between consecutive frames.
# ============================================================

print("")
print("=" * 70)
print("DAMAGE CHANGE ANALYSIS")
print("=" * 70)


predictions[
    "absolute_actual_damage_change"
] = (
    predictions[
        "current_to_future_change"
    ]
    .abs()
)


change_bins = [
    -np.inf,
    0.0005,
    0.0010,
    0.0025,
    0.0050,
    np.inf
]


change_labels = [
    "very_small",
    "small",
    "moderate",
    "large",
    "very_large"
]


predictions[
    "damage_change_level"
] = pd.cut(
    predictions[
        "absolute_actual_damage_change"
    ],
    bins=change_bins,
    labels=change_labels,
    include_lowest=True
)


change_records = []


for model in models:

    model_data = predictions[
        predictions[
            "model"
        ]
        ==
        model
    ]


    for level in change_labels:

        subset = model_data[
            model_data[
                "damage_change_level"
            ]
            ==
            level
        ]


        if len(
            subset
        ) == 0:

            continue


        change_records.append({

            "model":
                model,

            "damage_change_level":
                level,

            "records":
                int(
                    len(
                        subset
                    )
                ),

            "mean_actual_change":
                float(
                    subset[
                        "current_to_future_change"
                    ].mean()
                ),

            "mean_absolute_actual_change":
                float(
                    subset[
                        "absolute_actual_damage_change"
                    ].mean()
                ),

            "MAE":
                float(
                    subset[
                        "absolute_error"
                    ].mean()
                ),

            "RMSE":
                float(
                    np.sqrt(
                        subset[
                            "squared_error"
                        ].mean()
                    )
                )
        })


change_df = pd.DataFrame(
    change_records
)


# ============================================================
# SAVE ADDITIONAL CHANGE ANALYSIS
# ============================================================

OUTPUT_CHANGE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_1_damage_change_analysis.csv"
)


change_df.to_csv(
    OUTPUT_CHANGE,
    index=False
)


# ============================================================
# OVERALL INTERPRETATION
# ============================================================

persistence_row = model_summary[
    model_summary[
        "model"
    ]
    ==
    "Persistence"
]


persistence_mae = float(
    persistence_row.iloc[
        0
    ][
        "MAE"
    ]
)


comparison_summary_records_final = (
    comparison_summary
    .copy()
)


best_ml_model = None
best_ml_mae = None


ml_only = model_summary[
    model_summary[
        "model"
    ]
    !=
    "Persistence"
].copy()


if len(
    ml_only
) > 0:

    best_ml_row = ml_only.loc[
        ml_only[
            "MAE"
        ].idxmin()
    ]


    best_ml_model = (
        best_ml_row[
            "model"
        ]
    )


    best_ml_mae = float(
        best_ml_row[
            "MAE"
        ]
    )


if best_ml_mae is not None:

    ml_vs_persistence_percent = (
        (
            persistence_mae
            -
            best_ml_mae
        )
        /
        persistence_mae
    ) * 100.0

else:

    ml_vs_persistence_percent = 0.0


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = {

    "stage":
        "2.1",

    "title":
        "Baseline Error and Residual Analysis",

    "status":
        "PASS",

    "purpose":
        "Analyze Stage 2.0 predictions without retraining models.",

    "models":
        models,

    "model_summary":
        model_summary.to_dict(
            orient="records"
        ),

    "best_ml_model_by_overall_MAE":
        best_ml_model,

    "best_ml_MAE":
        best_ml_mae,

    "persistence_MAE":
        persistence_mae,

    "best_ML_vs_persistence_MAE_improvement_percent":
        float(
            ml_vs_persistence_percent
        ),

    "sequence_ranking":
        sequence_ranking_df.to_dict(
            orient="records"
        ),

    "model_vs_persistence":
        comparison_summary.to_dict(
            orient="records"
        ),

    "validation_note":
        "Current damage bins are analysis groups only and are not severity labels.",

    "outputs": {

        "sequence_analysis":
            OUTPUT_SEQUENCE,

        "component_analysis":
            OUTPUT_COMPONENT,

        "damage_level_analysis":
            OUTPUT_DAMAGE_BIN,

        "model_comparison":
            OUTPUT_COMPARISON,

        "prediction_cases":
            OUTPUT_CASES,

        "damage_change_analysis":
            OUTPUT_CHANGE,

        "summary":
            OUTPUT_JSON
    }
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.1 SUMMARY")
print("=" * 70)

print("")

print(
    "Models analyzed:",
    len(
        models
    )
)

print(
    "Sequences analyzed:",
    predictions[
        "sequence"
    ].nunique()
)

print(
    "Components analyzed:",
    predictions[
        "component"
    ].nunique()
)

print("")

print(
    "Persistence overall MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Best ML model by overall MAE:",
    best_ml_model
)

if best_ml_mae is not None:

    print(
        "Best ML overall MAE:",
        f"{best_ml_mae:.8f}"
    )

    print(
        "Best ML vs persistence MAE improvement:",
        f"{ml_vs_persistence_percent:.2f}%"
    )

print("")

print(
    "Sequence analysis:"
)

print(
    OUTPUT_SEQUENCE
)

print("")

print(
    "Component analysis:"
)

print(
    OUTPUT_COMPONENT
)

print("")

print(
    "Damage-level analysis:"
)

print(
    OUTPUT_DAMAGE_BIN
)

print("")

print(
    "Model comparison:"
)

print(
    OUTPUT_COMPARISON
)

print("")

print(
    "Difficult prediction cases:"
)

print(
    OUTPUT_CASES
)

print("")

print(
    "Damage-change analysis:"
)

print(
    OUTPUT_CHANGE
)

print("")

print(
    "Summary:"
)

print(
    OUTPUT_JSON
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")