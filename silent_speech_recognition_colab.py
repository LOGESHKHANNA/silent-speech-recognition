# %% [markdown]
# Silent Speech Recognition: Google Colab pipeline
#
# This notebook-style Python file follows the workflow in the PBL report:
# video input -> face/lip detection -> mouth ROI -> preprocessing -> CNN-LSTM
# -> speech class and measured evaluation.
#
# The report does not specify a dataset, vocabulary, sample count, speaker
# split, or trained weights. Set DATA_ROOT below to the actual dataset before
# running this file. No result value is assumed by this code.

# %% [markdown]
# Cell 1: install dependencies
#
# Run this once in a Google Colab cell before importing the rest of the file:
#
# !pip -q install mediapipe opencv-python-headless scikit-learn pandas matplotlib seaborn tqdm

# %% [markdown]
# Cell 2: imports and project configuration

from pathlib import Path
from typing import Optional, Tuple
import json
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm


# Mount Google Drive in Colab if the dataset is stored there.
try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass


SEED = 42
DATA_ROOT = Path("/content/drive/MyDrive/silent_speech_dataset")

# Optional manifest CSV. Leave as None when using class folders.
# A manifest must contain at least: video_path, label.
MANIFEST_PATH: Optional[Path] = None

OUTPUT_DIR = Path("/content/ssr_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These are implementation settings for this baseline, not reported results.
SEQ_LEN = 24
IMG_SIZE = (96, 96)  # width, height
MIN_VALID_FRAME_RATIO = 0.60
TEST_SIZE = 0.20
VAL_SIZE = 0.20
BATCH_SIZE = 4
EPOCHS = 30
USE_CACHE = True
CACHE_FILE = OUTPUT_DIR / "preprocessed_sequences.npz"

# Set this only when the manifest has a speaker column. For the folder layout
# root/class/speaker/video, the code can infer the speaker folder as "speaker".
SPEAKER_COLUMN: Optional[str] = "speaker"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
LIP_LANDMARK_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311,
    310, 415,
]


def set_seed(seed: int = SEED) -> None:
    """Make data splitting and model initialization repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed()
print(f"TensorFlow: {tf.__version__}")
print(f"MediaPipe: {mp.__version__}")
print(f"Dataset root: {DATA_ROOT}")

# %% [markdown]
# Cell 3: build or load a video manifest
#
# Supported folder layout:
#
# dataset/
#   CLASS_A/
#     video_001.mp4
#   CLASS_B/
#     video_002.mp4
#
# A speaker-aware layout is also supported:
#
# dataset/
#   CLASS_A/
#     SPEAKER_01/video_001.mp4
#
# The first directory below DATA_ROOT is treated as the class label. Adjust
# label_from_path() if the real dataset uses another layout.


def video_metadata(path: Path) -> dict:
    """Read lightweight metadata and confirm that OpenCV can open the file."""

    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    metadata = {
        "readable": bool(opened),
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
    }
    if opened:
        metadata["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        metadata["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        metadata["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        metadata["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = capture.read()
        metadata["readable"] = bool(ok and frame is not None and frame.size > 0)
    capture.release()
    return metadata


def label_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(f"Video is not inside a class folder: {path}")
    return relative.parts[0]


def inferred_speaker_from_path(path: Path, root: Path) -> Optional[str]:
    relative = path.relative_to(root)
    return relative.parts[1] if len(relative.parts) >= 3 else None


def build_folder_manifest(root: Path) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root}\n"
            "Update DATA_ROOT to the folder containing the class directories."
        )

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        metadata = video_metadata(path)
        rows.append(
            {
                "video_path": str(path),
                "label": label_from_path(path, root),
                "speaker": inferred_speaker_from_path(path, root),
                **metadata,
            }
        )

    if not rows:
        raise ValueError(f"No supported video files found below {root}")
    return pd.DataFrame(rows)


def load_csv_manifest(path: Path, root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"video_path", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    manifest = manifest.copy()

    def resolve_video_path(value: str) -> str:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root / candidate
        return str(candidate)

    manifest["video_path"] = manifest["video_path"].map(resolve_video_path)
    if "speaker" not in manifest.columns:
        manifest["speaker"] = None
    metadata_rows = [video_metadata(Path(path)) for path in manifest["video_path"]]
    metadata = pd.DataFrame(metadata_rows)
    return pd.concat([manifest.reset_index(drop=True), metadata], axis=1)


def clean_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Remove empty labels, missing files, duplicates, and unreadable videos."""

    cleaned = manifest.copy()
    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned = cleaned[~cleaned["label"].isin({"", "nan", "None"})]
    cleaned["video_path"] = cleaned["video_path"].map(lambda value: str(Path(value)))
    cleaned = cleaned.drop_duplicates(subset=["video_path"]).reset_index(drop=True)
    cleaned = cleaned[cleaned["video_path"].map(lambda value: Path(value).exists())]

    readable_rows = []
    for _, row in tqdm(cleaned.iterrows(), total=len(cleaned), desc="Validating videos"):
        metadata = video_metadata(Path(row["video_path"]))
        if metadata["readable"]:
            readable_rows.append(row)

    if not readable_rows:
        raise ValueError("No readable videos remain after cleaning.")

    cleaned = pd.DataFrame(readable_rows).reset_index(drop=True)
    return cleaned


if MANIFEST_PATH is not None:
    manifest = load_csv_manifest(MANIFEST_PATH, DATA_ROOT)
else:
    manifest = build_folder_manifest(DATA_ROOT)

manifest = clean_manifest(manifest)
manifest.to_csv(OUTPUT_DIR / "clean_manifest.csv", index=False)

print(f"Usable videos: {len(manifest)}")
print(f"Classes: {manifest['label'].nunique()}")
if "speaker" in manifest.columns:
    known_speakers = manifest["speaker"].dropna().astype(str)
    if not known_speakers.empty:
        print(f"Speakers in manifest: {known_speakers.nunique()}")
display(manifest.groupby("label").size().rename("video_count").to_frame())
display(manifest.head())

# %% [markdown]
# Cell 4: inspect the data and visualize raw frames


def show_raw_samples(frame: pd.DataFrame, samples: int = 6) -> None:
    sample = frame.sample(min(samples, len(frame)), random_state=SEED)
    columns = 3
    rows = int(np.ceil(len(sample) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (_, item) in zip(axes, sample.iterrows()):
        capture = cv2.VideoCapture(item["video_path"])
        ok, image = capture.read()
        capture.release()
        if ok:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{item['label']}\n{Path(item['video_path']).name}")
        axis.axis("off")
    for axis in axes[len(sample):]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


show_raw_samples(manifest)

plt.figure(figsize=(12, 4))
sns.countplot(data=manifest, x="label", order=sorted(manifest["label"].unique()))
plt.title("Samples per class before preprocessing")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# Cell 5: split videos into train, validation, and test sets
#
# Speaker leakage matters for visual speech. When a valid speaker column with
# at least three speakers is available, GroupShuffleSplit keeps speakers out
# of the other splits. Otherwise the code uses a stratified video split and
# prints a warning so the limitation is documented.


def choose_speaker_column(frame: pd.DataFrame) -> Optional[str]:
    if not SPEAKER_COLUMN or SPEAKER_COLUMN not in frame.columns:
        return None
    values = frame[SPEAKER_COLUMN].dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.nunique() < 3:
        return None
    return SPEAKER_COLUMN


def safe_stratify(frame: pd.DataFrame) -> Optional[pd.Series]:
    counts = frame["label"].value_counts()
    return frame["label"] if len(counts) and counts.min() >= 2 else None


def split_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    speaker_column = choose_speaker_column(frame)
    if speaker_column:
        groups = frame[speaker_column].astype(str).to_numpy()
        first_split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        train_val_indices, test_indices = next(first_split.split(frame, groups=groups))
        train_val = frame.iloc[train_val_indices].copy()
        test = frame.iloc[test_indices].copy()
        validation_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
        second_split = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=SEED)
        train_indices, validation_indices = next(
            second_split.split(train_val, groups=train_val[speaker_column].astype(str).to_numpy())
        )
        train = train_val.iloc[train_indices].copy()
        validation = train_val.iloc[validation_indices].copy()
        print(f"Speaker-aware split used with column: {speaker_column}")
    else:
        warnings.warn(
            "No usable speaker column was found. The split is video-stratified, "
            "so speaker leakage must be reported as a limitation."
        )
        train_val, test = train_test_split(
            frame,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=safe_stratify(frame),
        )
        validation_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
        train, validation = train_test_split(
            train_val,
            test_size=validation_fraction,
            random_state=SEED,
            stratify=safe_stratify(train_val),
        )

    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")
    split = pd.concat([train, validation, test], ignore_index=True)
    missing_from_train = set(split["label"]) - set(train["label"])
    if missing_from_train:
        raise ValueError(
            "These classes are absent from the training split: "
            f"{sorted(missing_from_train)}. Add more samples/speakers or adjust the split."
        )
    return split.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


split_manifest_df = split_manifest(manifest)
split_manifest_df.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
display(pd.crosstab(split_manifest_df["label"], split_manifest_df["split"]))

# %% [markdown]
# Cell 6: MediaPipe lip ROI extraction and frame preprocessing


mp_face_mesh = mp.solutions.face_mesh


def mouth_roi_from_landmarks(frame: np.ndarray, landmarks) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    points = [landmarks.landmark[index] for index in LIP_LANDMARK_INDICES]
    x_values = np.array([point.x * width for point in points])
    y_values = np.array([point.y * height for point in points])
    min_x, max_x = float(x_values.min()), float(x_values.max())
    min_y, max_y = float(y_values.min()), float(y_values.max())
    pad_x = max(18.0, (max_x - min_x) * 0.24)
    pad_y = max(14.0, (max_y - min_y) * 0.70)
    x1 = max(0, int(min_x - pad_x))
    y1 = max(0, int(min_y - pad_y))
    x2 = min(width, int(max_x + pad_x))
    y2 = min(height, int(max_y + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def sampled_frame_indices(frame_count: int, sequence_length: int) -> np.ndarray:
    if frame_count <= 0:
        return np.arange(sequence_length, dtype=np.int64)
    return np.linspace(0, frame_count - 1, sequence_length).round().astype(np.int64)


def extract_video_sequence(
    video_path: str,
    face_mesh,
    sequence_length: int = SEQ_LEN,
    image_size: Tuple[int, int] = IMG_SIZE,
    min_valid_ratio: float = MIN_VALID_FRAME_RATIO,
) -> Tuple[Optional[np.ndarray], float]:
    """Return a normalized fixed-length mouth ROI sequence and valid ratio."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None, 0.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sampled_frame_indices(frame_count, sequence_length)
    crops = []
    valid_indices = []

    if frame_count > 0:
        for position, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                crops.append(None)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            crop = None
            if result.multi_face_landmarks:
                crop = mouth_roi_from_landmarks(frame, result.multi_face_landmarks[0])
            if crop is None:
                crops.append(None)
                continue
            crop = cv2.resize(crop, image_size, interpolation=cv2.INTER_AREA)
            crops.append(crop)
            valid_indices.append(position)
    else:
        all_frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            all_frames.append(frame)
        if all_frames:
            fallback_indices = np.linspace(0, len(all_frames) - 1, sequence_length).round().astype(int)
            for position, frame_index in enumerate(fallback_indices):
                frame = all_frames[frame_index]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                crop = None
                if result.multi_face_landmarks:
                    crop = mouth_roi_from_landmarks(frame, result.multi_face_landmarks[0])
                if crop is None:
                    crops.append(None)
                    continue
                crops.append(cv2.resize(crop, image_size, interpolation=cv2.INTER_AREA))
                valid_indices.append(position)

    capture.release()
    valid_ratio = len(valid_indices) / max(sequence_length, 1)
    if not valid_indices or valid_ratio < min_valid_ratio:
        return None, valid_ratio

    # Fill occasional missed detections with the nearest valid crop. A video
    # is still discarded when too many sampled frames have no face/lip ROI.
    for position, crop in enumerate(crops):
        if crop is None:
            nearest = min(valid_indices, key=lambda index: abs(index - position))
            crops[position] = crops[nearest]

    sequence = np.stack(crops).astype(np.float32) / 255.0
    return sequence, valid_ratio


def preprocess_dataset(frame: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    sequences = []
    records = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Extracting mouth ROIs"):
            sequence, valid_ratio = extract_video_sequence(row["video_path"], face_mesh)
            if sequence is None:
                print(f"Skipping low-quality video: {row['video_path']} ({valid_ratio:.2f} valid frames)")
                continue
            sequences.append(sequence)
            records.append(
                {
                    "video_path": row["video_path"],
                    "label": row["label"],
                    "split": row["split"],
                    "valid_frame_ratio": valid_ratio,
                }
            )

    if not sequences:
        raise ValueError("No video produced a usable mouth ROI sequence.")
    return np.stack(sequences), pd.DataFrame(records)


if USE_CACHE and CACHE_FILE.exists():
    cached = np.load(CACHE_FILE, allow_pickle=True)
    X = cached["X"].astype(np.float32)
    processed_df = pd.DataFrame(
        {
            "video_path": cached["video_path"].astype(str),
            "label": cached["label"].astype(str),
            "split": cached["split"].astype(str),
            "valid_frame_ratio": cached["valid_frame_ratio"].astype(float),
        }
    )
    print(f"Loaded cached sequences: {X.shape}")
else:
    X, processed_df = preprocess_dataset(split_manifest_df)
    np.savez_compressed(
        CACHE_FILE,
        X=X,
        video_path=processed_df["video_path"].to_numpy(dtype=str),
        label=processed_df["label"].to_numpy(dtype=str),
        split=processed_df["split"].to_numpy(dtype=str),
        valid_frame_ratio=processed_df["valid_frame_ratio"].to_numpy(dtype=np.float32),
    )
    processed_df.to_csv(OUTPUT_DIR / "processed_manifest.csv", index=False)

print(f"Preprocessed tensor shape: {X.shape}")
display(processed_df.groupby(["split", "label"]).size().rename("sequence_count").to_frame())

# %% [markdown]
# Cell 7: inspect preprocessed mouth sequences


def show_processed_sequence(sequence: np.ndarray, label: str, frames: int = 6) -> None:
    indices = np.linspace(0, len(sequence) - 1, min(frames, len(sequence))).round().astype(int)
    figure, axes = plt.subplots(1, len(indices), figsize=(18, 3))
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, indices):
        axis.imshow(sequence[index])
        axis.set_title(f"frame {index}")
        axis.axis("off")
    figure.suptitle(f"Mouth ROI sequence: {label}")
    plt.tight_layout()
    plt.show()


show_processed_sequence(X[0], processed_df.iloc[0]["label"])

# %% [markdown]
# Cell 8: encode labels and create TensorFlow datasets


train_mask = processed_df["split"].eq("train").to_numpy()
validation_mask = processed_df["split"].eq("validation").to_numpy()
test_mask = processed_df["split"].eq("test").to_numpy()

if not train_mask.any() or not validation_mask.any() or not test_mask.any():
    raise ValueError("Train, validation, and test splits must all contain videos.")

label_encoder = LabelEncoder()
label_encoder.fit(processed_df.loc[train_mask, "label"])
class_names = label_encoder.classes_.tolist()
unknown_test_labels = set(processed_df.loc[~train_mask, "label"]) - set(class_names)
if unknown_test_labels:
    raise ValueError(f"Labels absent from training data: {sorted(unknown_test_labels)}")

y_all = label_encoder.transform(processed_df["label"])
X_train, y_train = X[train_mask], y_all[train_mask]
X_validation, y_validation = X[validation_mask], y_all[validation_mask]
X_test, y_test = X[test_mask], y_all[test_mask]
num_classes = len(class_names)

print(f"Classes used by the model: {class_names}")
print(f"Train: {X_train.shape}, validation: {X_validation.shape}, test: {X_test.shape}")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, tf.keras.utils.to_categorical(y_train, num_classes)))
    .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = (
    tf.data.Dataset.from_tensor_slices((X_validation, tf.keras.utils.to_categorical(y_validation, num_classes)))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Balanced weights are calculated from the real training labels.
class_values = np.unique(y_train)
class_values_weights = compute_class_weight("balanced", classes=class_values, y=y_train)
class_weight = dict(zip(class_values.tolist(), class_values_weights.tolist()))
print("Class weights:", class_weight)

# %% [markdown]
# Cell 9: build the CNN-LSTM model


def build_cnn_lstm(input_shape: tuple, number_of_classes: int, use_augmentation: bool = True) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="mouth_video")
    x = inputs

    if use_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomTranslation(0.04, 0.04),
                tf.keras.layers.RandomZoom(0.08),
                tf.keras.layers.RandomContrast(0.10),
            ],
            name="frame_augmentation",
        )
        x = tf.keras.layers.TimeDistributed(augmentation, name="augment_each_frame")(x)

    time_distributed = tf.keras.layers.TimeDistributed
    x = time_distributed(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))(x)
    x = time_distributed(tf.keras.layers.MaxPooling2D(2))(x)
    x = time_distributed(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))(x)
    x = time_distributed(tf.keras.layers.MaxPooling2D(2))(x)
    x = time_distributed(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"))(x)
    x = time_distributed(tf.keras.layers.GlobalAveragePooling2D(), name="spatial_features")(x)
    x = tf.keras.layers.LSTM(128, dropout=0.20, name="temporal_features")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax", name="speech_class")(x)

    model = tf.keras.Model(inputs, outputs, name="silent_speech_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


input_shape = (SEQ_LEN, IMG_SIZE[1], IMG_SIZE[0], 3)
model = build_cnn_lstm(input_shape, num_classes)
model.summary()

# %% [markdown]
# Cell 10: train and save the model


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_cnn_lstm.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

history_frame = pd.DataFrame(history.history)
figure, axes = plt.subplots(1, 2, figsize=(14, 4))
history_frame[["loss", "val_loss"]].plot(ax=axes[0], title="Training and validation loss")
history_frame[["accuracy", "val_accuracy"]].plot(ax=axes[1], title="Training and validation accuracy")
plt.tight_layout()
plt.show()

# %% [markdown]
# Cell 11: evaluate with the report metrics
#
# These values are calculated from the actual held-out test split. The report
# supplied with the project contains no measured values, so this cell is the
# place where real accuracy, precision, recall, and F1-score are produced.


test_probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
test_predictions = np.argmax(test_probabilities, axis=1)

metrics = {
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision_weighted": float(precision_score(y_test, test_predictions, average="weighted", zero_division=0)),
    "recall_weighted": float(recall_score(y_test, test_predictions, average="weighted", zero_division=0)),
    "f1_weighted": float(f1_score(y_test, test_predictions, average="weighted", zero_division=0)),
    "precision_macro": float(precision_score(y_test, test_predictions, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_test, test_predictions, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_test, test_predictions, average="macro", zero_division=0)),
}

print(json.dumps(metrics, indent=2))
print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0,
    )
)

with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

matrix = confusion_matrix(y_test, test_predictions, labels=np.arange(num_classes))
plt.figure(figsize=(max(7, len(class_names)), max(5, len(class_names) * 0.7)))
sns.heatmap(matrix, annot=True, fmt="d", cmap="mako", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Held-out test confusion matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# Cell 12: run inference on one video


def predict_video(video_path: str, trained_model: tf.keras.Model = model) -> dict:
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        sequence, valid_ratio = extract_video_sequence(video_path, face_mesh)

    if sequence is None:
        raise ValueError(
            f"Could not extract a mouth sequence from {video_path}. "
            f"Valid frame ratio: {valid_ratio:.2f}"
        )

    probabilities = trained_model.predict(sequence[None, ...], verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "video": str(video_path),
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "valid_frame_ratio": float(valid_ratio),
        "all_class_probabilities": {
            class_names[index]: float(probabilities[index])
            for index in range(len(class_names))
        },
    }


# Example after training:
# result = predict_video("/content/example.mp4")
# print(json.dumps(result, indent=2))

# %% [markdown]
# Notes for the project report
#
# 1. Record the actual dataset source, number of videos, classes, speakers, and
#    train/validation/test counts from the printed tables.
# 2. Record the real hyperparameters used: sequence length, image size, batch
#    size, epochs, learning rate, and the speaker split decision.
# 3. Copy evaluation_metrics.json values into the website only after testing.
# 4. Do not report values from the example configuration as experimental facts.
