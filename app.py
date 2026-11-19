import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from supabase import Client, create_client

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "satellite_model.tflite"
LABELS_PATH = BASE_DIR / "labels.json"

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
BUCKET_NAME = (os.getenv("SUPABASE_BUCKET") or "satellite-patches").strip()

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment variables/secrets."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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


def _predict_and_log(pil_img: Image.Image, filename: str):
    rgb_img = pil_img.convert("RGB")
    resized = rgb_img.resize((64, 64))

    prob_dict, top_label, top_conf = _run_tflite(resized)
    storage_path = _save_to_supabase(resized, filename, top_label, top_conf, prob_dict)

    status = (
        f"**Predicted:** `{top_label}` (**{top_conf * 100:.2f}%**)  \n"
        f"**Stored in bucket `{BUCKET_NAME}`:** `{storage_path}`  \n"
        f"**Logged to table:** `uploaded_patches`"
    )
    return resized, prob_dict, status


def classify_patch(pil_img: Image.Image | None, filename: str = "uploaded_patch.jpg"):
    if pil_img is None:
        return None, {}, "Please upload a satellite patch image to classify."
    return _predict_and_log(pil_img, filename)


def fetch_table_dataframe(table_name: str):
    if table_name not in {"uploaded_patches", "training_data", "test_data"}:
        return gr.update(value=[], headers=[])

    order_col = "created_at" if table_name == "uploaded_patches" else "id"
    res = (
        supabase.table(table_name)
        .select("*")
        .order(order_col, desc=True)
        .limit(20)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return gr.update(value=[], headers=["status"])

    hidden_cols = {"all_probabilities", "storage_path", "user_id"}
    headers = [col for col in rows[0].keys() if col not in hidden_cols]
    values = [[row.get(col) for col in headers] for row in rows]
    return gr.update(value=values, headers=headers)


with gr.Blocks(title="EuroSAT Satellite Patch Classifier") as demo:
    gr.Markdown(
        "# EuroSAT Satellite Patch Classifier\n"
        "MobileNetV2 (TFLite) land-cover classification backed by Supabase Storage & Postgres."
    )

    with gr.Row():
        with gr.Column():
            patch_input = gr.Image(type="pil", label="Satellite Patch (64x64 RGB)")
            classify_btn = gr.Button("Classify Patch", variant="primary")
            status_box = gr.Markdown("Ready.")

        with gr.Column():
            probs_output = gr.Label(num_top_classes=10, label="Class Probabilities")

    classify_btn.click(
        fn=classify_patch,
        inputs=[patch_input],
        outputs=[patch_input, probs_output, status_box],
    )

    gr.Markdown("## Supabase Database Explorer")
    with gr.Row():
        table_choice = gr.Radio(
            choices=["uploaded_patches", "training_data", "test_data"],
            value="uploaded_patches",
            label="Table",
        )
        refresh_btn = gr.Button("Refresh Table")

    table_display = gr.Dataframe(
        value=fetch_table_dataframe("uploaded_patches")["value"],
        headers=fetch_table_dataframe("uploaded_patches")["headers"],
        interactive=False,
    )

    table_choice.change(
        fn=fetch_table_dataframe,
        inputs=[table_choice],
        outputs=[table_display],
    )
    refresh_btn.click(
        fn=fetch_table_dataframe,
        inputs=[table_choice],
        outputs=[table_display],
    )

if __name__ == "__main__":
    demo.launch()