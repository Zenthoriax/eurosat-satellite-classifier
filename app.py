import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
from supabase import Client, create_client

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "satellite_model.tflite"
LABELS_PATH = BASE_DIR / "labels.json"

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
BUCKET_NAME = (os.getenv("SUPABASE_BUCKET") or "satellite-patches").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment variables."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)


def _run_tflite(resized_img: Image.Image) -> tuple[dict[str, float], str, float]:
    arr = np.asarray(resized_img, dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    top_idx = int(np.argmax(probs))
    top_label = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])
    return prob_dict, top_label, top_conf


def _save_to_supabase(
    resized_img: Image.Image,
    original_filename: str,
    top_label: str,
    top_conf: float,
    prob_dict: dict[str, float],
) -> str:
    jpg_buf = io.BytesIO()
    resized_img.save(jpg_buf, format="JPEG", quality=90)
    jpg_bytes = jpg_buf.getvalue()

    safe_stem = Path(original_filename or "patch").stem.replace(" ", "_")[:32]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unique_id = uuid.uuid4().hex[:8]
    storage_path = f"uploads/{stamp}_{unique_id}_{safe_stem}.jpg"
    record_filename = f"{stamp}_{unique_id}_{safe_stem}.jpg"

    supabase.storage.from_(BUCKET_NAME).upload(
        path=storage_path,
        file=jpg_bytes,
        file_options={"content-type": "image/jpeg"},
    )

    supabase.table("uploaded_patches").insert(
        {
            "filename": record_filename,
            "storage_path": storage_path,
            "predicted_label": top_label,
            "confidence": round(top_conf, 4),
            "all_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        }
    ).execute()

    return storage_path


@app.get("/")
def index():
    return render_template("index.html", bucket_name=BUCKET_NAME)


@app.post("/api/classify")
def api_classify():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "Please upload a satellite patch image."}), 400

    pil_img = Image.open(file.stream).convert("RGB")
    resized = pil_img.resize((64, 64))

    prob_dict, top_label, top_conf = _run_tflite(resized)
    storage_path = _save_to_supabase(resized, file.filename, top_label, top_conf, prob_dict)

    sorted_probs = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    return jsonify(
        {
            "predicted_label": top_label,
            "confidence": round(top_conf, 4),
            "confidence_pct": round(top_conf * 100, 2),
            "storage_path": storage_path,
            "bucket": BUCKET_NAME,
            "probabilities": [{"label": k, "score": round(v, 4)} for k, v in sorted_probs],
        }
    )


@app.get("/api/table/<table_name>")
def api_table(table_name: str):
    if table_name not in {"uploaded_patches", "training_data", "test_data"}:
        return jsonify({"error": "Invalid table name."}), 400

    order_col = "created_at" if table_name == "uploaded_patches" else "id"
    res = (
        supabase.table(table_name)
        .select("*")
        .order(order_col, desc=True)
        .limit(20)
        .execute()
    )
    rows = res.data or []
    hidden_cols = {"all_probabilities", "storage_path", "user_id"}
    headers = [col for col in rows[0].keys() if col not in hidden_cols] if rows else []
    clean_rows = [{k: row.get(k) for k in headers} for row in rows]
    return jsonify({"headers": headers, "rows": clean_rows})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)    