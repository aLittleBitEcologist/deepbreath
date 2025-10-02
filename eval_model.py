# eval_model.py

# python eval_model.py --val_normal_dir data/val/normal --val_pathology_dir data/val/pathology --test_normal_dir data/test/normal --test_pathology_dir data/test/pathology --artifacts_dir artifacts --img_size 256

import os, json, argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_utils import list_dicom_files, load_single_dicom
from metrics_utils import (
    compute_metrics,
    ecdf_fit,
    prob_normal_from_error,
)

def build_autoencoder(img_size=256, latent_dim=128):
    inp = tf.keras.Input((img_size, img_size, 1))
    x = inp
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(latent_dim, activation="relu")(x)
    x = tf.keras.layers.Dense((img_size//8)*(img_size//8)*128, activation="relu")(z)
    x = tf.keras.layers.Reshape((img_size//8, img_size//8, 128))(x)
    for filters in [128, 64, 32]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    return model

def load_artifacts(artifacts_dir, img_size_cli, latent_dim=128):
    model_path = os.path.join(artifacts_dir, "model.keras")
    best_full_path = os.path.join(artifacts_dir, "best_model.keras")
    best_weights_path = os.path.join(artifacts_dir, "best_model.weights.h5")

    stats = {}
    stats_path = os.path.join(artifacts_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

    img_size = img_size_cli if img_size_cli is not None else int(stats.get("img_size", 256))
    saved_threshold = stats.get("threshold", None)

    if os.path.exists(model_path):
        print(f"[INFO] Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
    elif os.path.exists(best_full_path):
        print(f"[INFO] Loading model: {best_full_path}")
        model = tf.keras.models.load_model(best_full_path)
    elif os.path.exists(best_weights_path):
        print(f"[INFO] Loading weights: {best_weights_path}")
        model = build_autoencoder(img_size=img_size, latent_dim=latent_dim)
        model.load_weights(best_weights_path)
    else:
        raise FileNotFoundError(
            "Нет артефактов модели: "
            f"{model_path} | {best_full_path} | {best_weights_path}"
        )

    ecdf_path = os.path.join(artifacts_dir, "calibration_ecdf.npy")
    xs_sorted = np.load(ecdf_path) if os.path.exists(ecdf_path) else None

    return model, xs_sorted, img_size, saved_threshold

def eval_split_files(model, xs_sorted, img_size, normal_dir, pathology_dir):
    y_true, p_pathology = [], []
    for label, root in [(0, normal_dir), (1, pathology_dir)]:
        if root is None or (not os.path.isdir(root)):
            continue
        files = list_dicom_files(root)
        if len(files) == 0:
            print(f"[WARN] Нет DICOM-файлов в: {root}")
            continue
        for fp in tqdm(files, desc=f"Eval {os.path.basename(root)}"):
            try:
                x, _ = load_single_dicom(fp, img_size=img_size)
                x_hat = model.predict(x[None, ...], verbose=0)[0]
                err = float(np.mean(np.abs(x - x_hat)))
                p_norm = prob_normal_from_error(np.array([err], dtype=np.float32), xs_sorted)[0]
                p_path = float(1.0 - p_norm)

                if p_path < 0.0: p_path = 0.0
                if p_path > 1.0: p_path = 1.0
                y_true.append(label); p_pathology.append(p_path)
            except Exception as e:
                print(f"[WARN] Пропускаю файл {fp}: {type(e).__name__}: {e}")
                continue
    return np.array(y_true), np.array(p_pathology)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--val_normal_dir", required=True)
    ap.add_argument("--val_pathology_dir", required=True)
    ap.add_argument("--test_normal_dir", required=False, default=None)
    ap.add_argument("--test_pathology_dir", required=False, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)  # CLI > saved > 0.5
    ap.add_argument("--latent_dim", type=int, default=128)
    args = ap.parse_args()

    model, xs_sorted, img_size, saved_thr = load_artifacts(args.artifacts_dir, args.img_size, latent_dim=args.latent_dim)

    if xs_sorted is None:
        print("[INFO] calibration_ecdf.npy не найден — калибруюсь по val_normal_dir (по файлам)...")
        norm_files = list_dicom_files(args.val_normal_dir)
        per_file_errors = []
        for fp in tqdm(norm_files, desc="Calibrating ECDF on val-normal"):
            try:
                x, _ = load_single_dicom(fp, img_size=img_size)
                x_hat = model.predict(x[None, ...], verbose=0)[0]
                err = float(np.mean(np.abs(x - x_hat)))
                per_file_errors.append(err)
            except Exception as e:
                print(f"[WARN] ECDF skip {fp}: {e}")
        if len(per_file_errors) == 0:
            raise ValueError("Не смог посчитать ECDF: пустая валидация норм.")
        xs_sorted = ecdf_fit(np.array(per_file_errors, dtype=np.float32))

    if args.threshold is not None:
        thr = float(args.threshold)
        source = "CLI"
    elif saved_thr is not None:
        thr = float(saved_thr)
        source = "stats.json"
    else:
        thr = 0.5
        source = "default"

    thr_eff = float(thr)
    if thr_eff >= 1.0:
        thr_eff = float(np.nextafter(1.0, 0.0))
    elif thr_eff <= 0.0:
        thr_eff = float(np.nextafter(0.0, 1.0))
    print(f"[INFO] Using threshold from {source}: {thr:.6f}  |  effective: {thr_eff:.8f}")

    print("== Validation ==")
    yv, pv = eval_split_files(model, xs_sorted, img_size, args.val_normal_dir, args.val_pathology_dir)
    print(f"[INFO] Files on validation: {len(yv)} (normals={int(np.sum(yv==0))}, pathologies={int(np.sum(yv==1))})")
    mv = compute_metrics(yv, pv, threshold=thr_eff)
    for k, v in mv.items():
        print(f"{k}: {v}")

    if args.test_normal_dir or args.test_pathology_dir:
        print("\n== Test ==")
        yt, pt = eval_split_files(model, xs_sorted, img_size, args.test_normal_dir, args.test_pathology_dir)
        print(f"[INFO] Files on test: {len(yt)} (normals={int(np.sum(yt==0))}, pathologies={int(np.sum(yt==1))})")
        mt = compute_metrics(yt, pt, threshold=thr_eff)
        for k, v in mt.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
