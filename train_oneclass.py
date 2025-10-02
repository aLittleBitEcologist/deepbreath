# train_oneclass.py

# python train_oneclass.py --train_dir data/train/normal --val_normal_dir data/val/normal --val_pathology_dir data/val/pathology --out_dir artifacts --img_size 256 --batch_size 16 --epochs 30 --tune_threshold --tune_mode youden

import os, json, argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_utils import list_dicom_files, load_single_dicom
from metrics_utils import ecdf_fit, prob_normal_from_error, compute_metrics

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

def make_loader_fn(img_size: int):
    def _loader(path_bytes):

        if isinstance(path_bytes, (bytes, np.bytes_)):
            path = path_bytes.decode("utf-8")
        else:
            path = path_bytes
        x, _ = load_single_dicom(path, img_size=img_size)
        return x
    return _loader

def dataset_from_filelist(file_list, img_size, batch_size, shuffle=True):
    if len(file_list) == 0:
        raise ValueError("No data files provided.")
    paths = tf.constant(file_list, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    loader_fn = make_loader_fn(img_size)
    ds = ds.map(lambda p: tf.numpy_function(loader_fn, [p], Tout=tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.ensure_shape(x, [img_size, img_size, 1]),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(file_list), 10000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def find_best_threshold(y_true, p_pathology, mode="youden", spec_target=None):
    """
    y_true: {0,1}, p_pathology: [0..1]
    mode:
      - "youden": max(Sens + Spec - 1)
      - "acc":    max(Accuracy)
      - "spec_target": минимальный порог с Specificity >= spec_target
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p_pathology)

    thr_cands = np.unique(p)
    thr_cands = np.concatenate(([0.0], thr_cands, [1.0]))

    if mode == "spec_target":
        if spec_target is None:
            raise ValueError("spec_target must be set for mode='spec_target'")
        valid = []
        for t in thr_cands:
            m = compute_metrics(y_true, p, threshold=float(t))
            if m["Specificity"] >= spec_target:
                valid.append((float(t), m))
        if len(valid) == 0:

            return find_best_threshold(y_true, p, mode="youden")

        valid.sort(key=lambda tm: (-tm[1]["Sensitivity"], tm[0]))
        return float(valid[0][0])

    best_thr = 0.5
    best_score = -1e9
    for t in thr_cands:
        m = compute_metrics(y_true, p, threshold=float(t))
        score = m["Accuracy"] if mode == "acc" else (m["Sensitivity"] + m["Specificity"] - 1.0)
        if score > best_score:
            best_score = score
            best_thr = float(t)
    return best_thr

def safe_threshold(thr: float) -> float:
    if thr >= 1.0:
        return float(np.nextafter(1.0, 0.0))
    if thr <= 0.0:
        return float(np.nextafter(0.0, 1.0))
    return float(thr)

def file_recon_error(model, fp, img_size: int):
    x, _ = load_single_dicom(fp, img_size=img_size)      # [H,W,1]
    x_hat = model.predict(x[None, ...], verbose=0)[0]    # [H,W,1]
    err = float(np.mean(np.abs(x - x_hat)))
    return err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_normal_dir", required=True)
    ap.add_argument("--val_pathology_dir", default=None)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent_dim", type=int, default=128)

    ap.add_argument("--tune_threshold", action="store_true",
                    help="Подобрать порог по валидации и сохранить его в stats.json")
    ap.add_argument("--tune_mode", type=str, default="youden",
                    choices=["youden", "acc", "spec_target"],
                    help="Критерий подбора порога")
    ap.add_argument("--target_specificity", type=float, default=None,
                    help="Целевая специфичность для tune_mode=spec_target")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_autoencoder(args.img_size, args.latent_dim)
    model.summary()

    train_files = list_dicom_files(args.train_dir)
    val_norm_files = list_dicom_files(args.val_normal_dir)
    val_path_files = list_dicom_files(args.val_pathology_dir) if args.val_pathology_dir else []
    if len(train_files) == 0:
        raise ValueError(f"No DICOM files found in train_dir: {args.train_dir}")
    if len(val_norm_files) == 0:
        raise ValueError(f"No DICOM files found in val_normal_dir: {args.val_normal_dir}")
    print(f"[INFO] train files: {len(train_files)}  |  val-normal files: {len(val_norm_files)}  |  val-pathology files: {len(val_path_files)}")

    ds_train = dataset_from_filelist(train_files, args.img_size, args.batch_size, shuffle=True)
    ds_val   = dataset_from_filelist(val_norm_files, args.img_size, args.batch_size, shuffle=False)

    ckpt_path = os.path.join(args.out_dir, "best_model.keras")
    print("[DEBUG] Keras:", tf.keras.__version__)
    print("[DEBUG] Checkpoint path:", ckpt_path)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    model.fit(
        ds_train,
        epochs=args.epochs,
        validation_data=ds_val,
        callbacks=[ckpt, earlystop]
    )

    model_path = os.path.join(args.out_dir, "model.keras")
    model.save(model_path)
    model = tf.keras.models.load_model(model_path)

    per_file_errors_norm = []
    for fp in tqdm(val_norm_files, desc="Calibrating ECDF on val-normal"):
        try:
            per_file_errors_norm.append(file_recon_error(model, fp, args.img_size))
        except Exception as e:
            print(f"[WARN] ECDF skip {fp}: {e}")
    if len(per_file_errors_norm) == 0:
        raise ValueError("ECDF calibration failed: no valid val-normal files.")
    xs_sorted = ecdf_fit(np.array(per_file_errors_norm, dtype=np.float32))
    np.save(os.path.join(args.out_dir, "calibration_ecdf.npy"), xs_sorted)

    tuned_threshold_raw = None
    tuned_threshold_safe = None
    tuning_info = {}

    if args.tune_threshold:

        p_norm = []
        for err in per_file_errors_norm:
            p_norm.append(1.0 - prob_normal_from_error(np.array([err], dtype=np.float32), xs_sorted)[0])

        y_true = [0] * len(p_norm)
        p_path = p_norm.copy()

        if len(val_path_files) > 0:
            for fp in tqdm(val_path_files, desc="Scoring val-pathology"):
                try:
                    err = file_recon_error(model, fp, args.img_size)
                    p = 1.0 - prob_normal_from_error(np.array([err], dtype=np.float32), xs_sorted)[0]
                    y_true.append(1)
                    p_path.append(float(p))
                except Exception as e:
                    print(f"[WARN] Skip pathology {fp}: {e}")

        y_true = np.array(y_true, dtype=np.int32)
        p_path = np.array(p_path, dtype=np.float32)

        if (len(val_path_files) == 0) and (args.tune_mode == "spec_target") and (args.target_specificity is not None):
            q = max(0.0, min(1.0, 1.0 - float(args.target_specificity)))
            tuned_threshold_raw = float(np.quantile(p_path, q))
            print(f"[INFO] Tuned threshold from normals only: {tuned_threshold_raw:.6f} (target specificity={args.target_specificity})")
        else:
            tuned_threshold_raw = find_best_threshold(
                y_true, p_path, mode=args.tune_mode, spec_target=args.target_specificity
            )
            print(f"[INFO] Tuned threshold on validation (raw): {tuned_threshold_raw:.6f} (mode={args.tune_mode})")

        tuned_threshold_safe = safe_threshold(tuned_threshold_raw)
        if tuned_threshold_safe != tuned_threshold_raw:
            print(f"[INFO] Effective threshold saved/used: {tuned_threshold_safe:.8f} (clamped from raw {tuned_threshold_raw:.6f})")

        if len(val_path_files) > 0:
            mv = compute_metrics(y_true, p_path, threshold=tuned_threshold_safe)
            print("[INFO] Validation metrics at tuned threshold:")
            for k, v in mv.items():
                print(f"{k}: {v}")

        tuning_info = {
            "threshold": tuned_threshold_safe,
            "threshold_raw": tuned_threshold_raw,
            "tune_mode": args.tune_mode,
            "target_specificity": args.target_specificity,
            "val_counts": {"normal_files": int(len(val_norm_files)), "pathology_files": int(len(val_path_files))}
        }

    stats = {"img_size": args.img_size}
    if tuned_threshold_safe is not None:
        stats.update(tuning_info)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f)

    print("Training done. Artifacts saved to:", args.out_dir)

if __name__ == "__main__":
    main()
