"""
Evaluate satellite_model.tflite on the EuroSAT RGB held-out test split.

Reproduces the exact 80/20 stratified split used in train_and_seed.py
(random_state=42) and runs the *deployed* TFLite model with the same
preprocessing as app.py (RGB -> resize 64x64 -> /255 -> batch of 1).

Outputs (in --out-dir, default ./evaluation):
    metrics.json              all scalar + per-class metrics
    classification_report.txt sklearn text report
    confusion_matrix.csv      raw counts
    confusion_matrix.png      raw + row-normalized heatmaps
    predictions.csv           per-image predictions
    metrics_tables.tex        booktabs tables ready to \\input{} in LaTeX
    metrics.md                markdown tables ready to paste in README

Usage:
    pip install numpy pillow scikit-learn matplotlib tensorflow-cpu
    python evaluate.py --data dataset/EuroSAT_RGB

Dataset: https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

IMG_SIZE = (64, 64)


DATASET_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"


def download_dataset(data_dir: Path) -> None:
    """Download and extract EuroSAT RGB (~90 MB) into data_dir's parent, like train_and_seed.py."""
    import urllib.request
    import zipfile

    root = data_dir.parent if data_dir.name == "EuroSAT_RGB" else data_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "EuroSAT_RGB.zip"
    if not zip_path.exists():
        print(f"Downloading EuroSAT RGB (~90 MB) to {zip_path} ...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)


def find_images_root(data_dir: Path) -> Path:
    """Return the folder that directly contains the class subfolders of .jpg files."""
    candidates = [data_dir, data_dir / "EuroSAT_RGB", data_dir / "EuroSAT_RGB" / "EuroSAT_RGB"]
    for c in candidates:
        if c.is_dir():
            class_dirs = [d for d in c.iterdir() if d.is_dir() and any(d.glob("*.jpg"))]
            if len(class_dirs) >= 2:
                return c
    raise FileNotFoundError(f"No EuroSAT class folders with .jpg images found under: {data_dir.resolve()}")


def gather(images_root: Path):
    """Same ordering as train_and_seed.py: sorted classes, sorted *.jpg per class."""
    class_names = sorted(d.name for d in images_root.iterdir() if d.is_dir())
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        for p in sorted((images_root / cls).glob("*.jpg")):
            paths.append(str(p))
            labels.append(idx)
    return class_names, paths, labels


def load_input(path: str, dtype) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0).astype(dtype)


def plot_confusion(cm, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = format(data[i, j], fmt)
                ax.text(j, i, val, ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_latex(m, class_names, out_path):
    lines = []
    lines += [
        r"\begin{table}[h]", r"\centering",
        r"\begin{tabular}{@{}lr@{}}", r"\toprule",
        r"\textbf{Metric} & \textbf{Value}\\", r"\midrule",
        f"Test images & {m['n_test']:,}\\\\".replace(",", r"{,}"),
        f"Accuracy (top-1) & {m['accuracy']*100:.2f}\\%\\\\",
        f"Top-3 accuracy & {m['top3_accuracy']*100:.2f}\\%\\\\",
        f"Macro precision & {m['macro']['precision']*100:.2f}\\%\\\\",
        f"Macro recall & {m['macro']['recall']*100:.2f}\\%\\\\",
        f"Macro F1 & {m['macro']['f1']*100:.2f}\\%\\\\",
        f"Weighted F1 & {m['weighted']['f1']*100:.2f}\\%\\\\",
        f"Cohen's $\\kappa$ & {m['cohen_kappa']:.4f}\\\\",
        f"Matthews corr.\\ coef. & {m['matthews_corrcoef']:.4f}\\\\",
        f"Mean latency (ms/image, CPU, batch 1) & {m['latency_ms']['mean']:.2f}\\\\",
        f"Median latency (ms) & {m['latency_ms']['median']:.2f}\\\\",
        f"95th percentile latency (ms) & {m['latency_ms']['p95']:.2f}\\\\",
        f"Model size (MB) & {m['model_size_mb']:.3f}\\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Aggregate evaluation of \texttt{satellite\_model.tflite} on the held-out EuroSAT test split.}",
        r"\label{tab:eval-aggregate}", r"\end{table}", "",
        r"\begin{table}[h]", r"\centering",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support}\\",
        r"\midrule",
    ]
    for c in class_names:
        pc = m["per_class"][c]
        lines.append(
            f"{latex_escape(c)} & {pc['precision']:.4f} & {pc['recall']:.4f} & "
            f"{pc['f1']:.4f} & {pc['support']}\\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Per-class precision, recall, F1 and support.}",
        r"\label{tab:eval-per-class}", r"\end{table}", "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(m, class_names, out_path):
    L = [
        "| Metric | Value |", "|---|---|",
        f"| Test images | {m['n_test']:,} |",
        f"| Accuracy (top-1) | {m['accuracy']*100:.2f}% |",
        f"| Top-3 accuracy | {m['top3_accuracy']*100:.2f}% |",
        f"| Macro precision / recall / F1 | {m['macro']['precision']*100:.2f}% / "
        f"{m['macro']['recall']*100:.2f}% / {m['macro']['f1']*100:.2f}% |",
        f"| Weighted F1 | {m['weighted']['f1']*100:.2f}% |",
        f"| Cohen's kappa | {m['cohen_kappa']:.4f} |",
        f"| Matthews corr. coef. | {m['matthews_corrcoef']:.4f} |",
        f"| Latency, mean / median / p95 (ms/image) | {m['latency_ms']['mean']:.2f} / "
        f"{m['latency_ms']['median']:.2f} / {m['latency_ms']['p95']:.2f} |",
        f"| Model size | {m['model_size_mb']:.3f} MB |",
        "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for c in class_names:
        pc = m["per_class"][c]
        L.append(f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    L += ["", f"_Measured on: {m['environment']['platform']}, Python "
          f"{m['environment']['python']}. Latency depends on hardware._"]
    out_path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/EuroSAT_RGB", help="Extracted EuroSAT RGB folder")
    ap.add_argument("--model", default="satellite_model.tflite")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out-dir", default="evaluation")
    ap.add_argument("--warmup", type=int, default=20, help="Untimed warm-up invocations")
    ap.add_argument("--no-download", action="store_true",
                    help="Do not auto-download the dataset if it is missing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    try:
        images_root = find_images_root(data_dir)
    except FileNotFoundError:
        if args.no_download:
            raise
        download_dataset(data_dir)
        images_root = find_images_root(data_dir)
    class_names, paths, labels = gather(images_root)
    print(f"Found {len(class_names)} classes, {len(paths)} images")

    # Class order of the model comes from labels.json; it must match the folders.
    with open(args.labels, "r", encoding="utf-8") as f:
        model_labels = json.load(f)
    if model_labels != class_names:
        raise SystemExit(
            "labels.json does not match dataset folders.\n"
            f"labels.json: {model_labels}\nfolders:     {class_names}"
        )

    _, test_paths, _, test_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    y_true = np.array(test_labels)
    print(f"Test split: {len(test_paths)} images")

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Warm-up
    for p in test_paths[: args.warmup]:
        interpreter.set_tensor(inp["index"], load_input(p, inp["dtype"]))
        interpreter.invoke()

    probs_all = np.zeros((len(test_paths), len(class_names)), dtype=np.float32)
    lat_ms = []
    for i, p in enumerate(test_paths):
        x = load_input(p, inp["dtype"])
        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])[0]
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        probs_all[i] = probs
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(test_paths)}")

    y_pred = probs_all.argmax(axis=1)
    conf = probs_all.max(axis=1)
    k = len(class_names)
    all_idx = list(range(k))

    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=all_idx, zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_idx, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_idx, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_idx)
    correct = y_true == y_pred

    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probs_all, k=3, labels=all_idx)),
        "macro": {"precision": float(p_m), "recall": float(r_m), "f1": float(f_m)},
        "weighted": {"precision": float(p_w), "recall": float(r_w), "f1": float(f_w)},
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
        "per_class": {
            c: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f_c[i]),
                "support": int(s_c[i]),
                "accuracy_pct": float(cm[i, i] / max(cm[i].sum(), 1) * 100),
            }
            for i, c in enumerate(class_names)
        },
        "latency_ms": {
            "mean": float(statistics.fmean(lat_ms)),
            "median": float(statistics.median(lat_ms)),
            "p95": float(np.percentile(lat_ms, 95)),
            "p99": float(np.percentile(lat_ms, 99)),
            "min": float(min(lat_ms)),
            "max": float(max(lat_ms)),
            "note": "Interpreter set_tensor+invoke+get_tensor only, batch size 1, CPU; "
                    "excludes image decoding. Hardware dependent.",
        },
        "model_size_mb": Path(args.model).stat().st_size / (1024 * 1024),
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
    }

    # Most confused pairs (off-diagonal, top 10)
    pairs = []
    for i in range(k):
        for j in range(k):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    metrics["top_confusions"] = [
        {"count": c, "true": t, "predicted": p} for c, t, p in pairs[:10]
    ]

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=all_idx, target_names=class_names, digits=4, zero_division=0),
        encoding="utf-8",
    )
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "predicted_label", "confidence", "is_correct"])
        for pth, t, pr, c in zip(test_paths, y_true, y_pred, conf):
            w.writerow([Path(pth).name, class_names[t], class_names[pr],
                        round(float(c), 4), bool(t == pr)])

    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    write_latex(metrics, class_names, out_dir / "metrics_tables.tex")
    write_markdown(metrics, class_names, out_dir / "metrics.md")

    print("\n" + (out_dir / "classification_report.txt").read_text(encoding="utf-8"))
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%   Macro F1: {metrics['macro']['f1']*100:.2f}%")
    print(f"Latency mean/median/p95: {metrics['latency_ms']['mean']:.2f} / "
          f"{metrics['latency_ms']['median']:.2f} / {metrics['latency_ms']['p95']:.2f} ms")
    print(f"All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()