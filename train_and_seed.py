import os
import json
import zipfile
import requests
import numpy as np
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file.")

DATA_DIR = Path("dataset")
ZIP_PATH = DATA_DIR / "EuroSAT_RGB.zip"
EXTRACT_DIR = DATA_DIR / "EuroSAT_RGB"
DATASET_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

IMG_SIZE = (64, 64)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# 1. Download & Extract EuroSAT RGB
DATA_DIR.mkdir(exist_ok=True)
if not EXTRACT_DIR.exists():
    print("Downloading EuroSAT RGB dataset (~90 MB)...")
    r = requests.get(DATASET_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

# Handle nested folder if EuroSAT_RGB/EuroSAT_RGB exists
nested_dir = EXTRACT_DIR / "EuroSAT_RGB"
images_root = nested_dir if nested_dir.exists() else EXTRACT_DIR

# 2. Gather File Paths & Labels
class_names = sorted([d.name for d in images_root.iterdir() if d.is_dir()])
print(f"Found {len(class_names)} classes: {class_names}")

with open("labels.json", "w") as f:
    json.dump(class_names, f, indent=2)

file_paths, labels = [], []
for idx, cls in enumerate(class_names):
    for img_path in sorted((images_root / cls).glob("*.jpg")):
        file_paths.append(str(img_path))
        labels.append(idx)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. Build tf.data Pipelines
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(10000, seed=42)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Important: unshuffled test_ds so predictions align 1:1 with test_paths & test_labels
test_ds = (
    tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# 4. Build & Train Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3), include_top=False, weights="imagenet"
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Training for {EPOCHS} epochs...")
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# 5. Evaluate Test Set & Export TFLite Model
print("Running test set inference...")
preds = model.predict(test_ds, verbose=0)
pred_indices = np.argmax(preds, axis=1)
confidences = np.max(preds, axis=1)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("satellite_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Exported satellite_model.tflite and labels.json.")

# 6. Seed Supabase Tables
def batch_insert(table_name, rows, batch_size=500):
    for i in range(0, len(rows), batch_size):
        supabase.table(table_name).insert(rows[i:i + batch_size]).execute()
    print(f"Inserted {len(rows)} rows into '{table_name}'.")

train_rows = [
    {"filename": Path(p).name, "true_label": class_names[l], "split": "train"}
    for p, l in zip(train_paths, train_labels)
]

test_rows = [
    {
        "filename": Path(p).name,
        "true_label": class_names[true_l],
        "predicted_label": class_names[pred_l],
        "confidence": round(float(conf), 4),
        "is_correct": bool(true_l == pred_l)
    }
    for p, true_l, pred_l, conf in zip(test_paths, test_labels, pred_indices, confidences)
]

print("Syncing train and test data to Supabase...")
batch_insert("training_data", train_rows)
batch_insert("test_data", test_rows)
print("Done!")