# infer_zip.py

# python infer_zip.py --zip_paths data/test.zip --artifacts_dir artifacts --xlsx_out results.xlsx --img_size 256

# python infer_zip.py --zip_paths data/test_modality_bpe.zip --artifacts_dir artifacts --xlsx_out results.xlsx --img_size 256

import os, json, time, argparse, shutil, tempfile, logging, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from data_utils import load_single_dicom
from metrics_utils import prob_normal_from_error

try:
    import pydicom
    from pydicom.config import logger as pydicom_logger
    pydicom.config.enforce_valid_values = False
    warnings.filterwarnings("ignore", message="Invalid value for VR UI")
    pydicom_logger.setLevel(logging.ERROR)
except Exception:
    pydicom = None

REQUIRED_COLUMNS = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "processing_status",
    "time_of_processing",
]

def load_artifacts(artifacts_dir: str, img_size_cli: int | None):
    """Грузим модель, ECDF и метаданные (включая threshold)."""
    model_path     = os.path.join(artifacts_dir, "model.keras")
    best_full_path = os.path.join(artifacts_dir, "best_model.keras")
    ecdf_path      = os.path.join(artifacts_dir, "calibration_ecdf.npy")
    stats_path     = os.path.join(artifacts_dir, "stats.json")

    if os.path.exists(model_path):
        print(f"[INFO] Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
    elif os.path.exists(best_full_path):
        print(f"[INFO] Loading model: {best_full_path}")
        model = tf.keras.models.load_model(best_full_path)
    else:
        raise FileNotFoundError(
            f"Model not found in {artifacts_dir} (expected model.keras or best_model.keras)"
        )

    if not os.path.exists(ecdf_path):
        raise FileNotFoundError(f"ECDF calibration not found: {ecdf_path}")
    xs_sorted = np.load(ecdf_path)

    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

    img_size = img_size_cli if img_size_cli is not None else int(stats.get("img_size", 256))
    saved_threshold = stats.get("threshold", None)

    return model, xs_sorted, img_size, saved_threshold

def header_checks(fp: str):
    """
    Быстрая проверка заголовка DICOM без чтения пикселей.
    Требования: Modality == 'CT' и BodyPartExamined == 'CHEST'.
    Возвращает:
      ok: bool,
      reasons: list[str] (пустой список если всё ок),
      meta: dict(study_uid, series_uid, modality, body_part)
    """
    if pydicom is None:
        raise ImportError("pydicom is required for header checks")

    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    modality = str(getattr(ds, "Modality", "")).strip().upper()
    body     = str(getattr(ds, "BodyPartExamined", "")).strip().upper()
    study_uid  = str(getattr(ds, "StudyInstanceUID", "") or "")
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")

    reasons = []
    if modality != "CT":
        reasons.append(f"Modality!=CT ({modality or 'EMPTY'})")
    if body != "CHEST":
        reasons.append(f"BodyPartExamined!=CHEST ({body or 'EMPTY'})")

    ok = (len(reasons) == 0)
    meta = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "modality": modality,
        "body_part": body,
    }
    return ok, reasons, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_paths", nargs="+", required=True, help="Список ZIP-архивов для инференса")
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--xlsx_out", required=True)
    ap.add_argument("--img_size", type=int, default=None)

    ap.add_argument("--threshold", type=float, default=None,
                    help="Порог для бинаризации p(pathology); по умолчанию из stats.json/0.5")
    args = ap.parse_args()

    model, xs_sorted, img_size, saved_thr = load_artifacts(args.artifacts_dir, args.img_size)
    if args.threshold is not None:
        thr = float(args.threshold)
        print(f"[INFO] Using threshold from CLI: {thr:.6f}")
    elif saved_thr is not None:
        thr = float(saved_thr)
        print(f"[INFO] Using threshold from stats.json: {thr:.6f}")
    else:
        thr = 0.5
        print(f"[INFO] Using default threshold: {thr:.6f}")

    rows = []
    tmp_root = tempfile.mkdtemp(prefix="ct_zip_")

    try:
        for zp in args.zip_paths:
            zip_name = os.path.basename(zp)
            extract_dir = os.path.join(tmp_root, f"{zip_name}_extracted")
            start_zip = time.time()

            try:
                import zipfile
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                rows.append({
                    "path_to_study": f"{zip_name}::",
                    "study_uid": "",
                    "series_uid": "",
                    "probability_of_pathology": np.nan,
                    "pathology": "",
                    "processing_status": f"Failure: {type(e).__name__}: {e}",
                    "time_of_processing": time.time() - start_zip,
                })
                continue

            file_paths = []
            for r, _, fs in os.walk(extract_dir):
                for f in fs:
                    file_paths.append(os.path.join(r, f))
            file_paths.sort()

            if len(file_paths) == 0:
                rows.append({
                    "path_to_study": f"{zip_name}::",
                    "study_uid": "",
                    "series_uid": "",
                    "probability_of_pathology": np.nan,
                    "pathology": "",
                    "processing_status": "Failure: Empty archive",
                    "time_of_processing": time.time() - start_zip,
                })
                continue

            for fp in tqdm(file_paths, desc=f"Scoring {zip_name}", leave=False):
                t0 = time.time()
                inner_path = os.path.relpath(fp, extract_dir)
                path_to_study = f"{zip_name}::{inner_path}"

                try:
                    ok, reasons, meta_hdr = header_checks(fp)
                    suid = meta_hdr.get("study_uid", "")
                    srid = meta_hdr.get("series_uid", "")
                except Exception as e:
                    rows.append({
                        "path_to_study": path_to_study,
                        "study_uid": "",
                        "series_uid": "",
                        "probability_of_pathology": np.nan,
                        "pathology": "",
                        "processing_status": f"Failure: HeaderReadError: {type(e).__name__}: {e}",
                        "time_of_processing": time.time() - t0,
                    })
                    continue

                if not ok:

                    rows.append({
                        "path_to_study": path_to_study,
                        "study_uid": suid,
                        "series_uid": srid,
                        "probability_of_pathology": np.nan,
                        "pathology": "",
                        "processing_status": "Failure: " + ", ".join(reasons),
                        "time_of_processing": time.time() - t0,
                    })
                    continue

                try:
                    x, _ = load_single_dicom(fp, img_size=img_size)
                    x_hat = model.predict(x[None, ...], verbose=0)[0]
                    err = float(np.mean(np.abs(x - x_hat)))
                    p_path = float(1.0 - prob_normal_from_error(np.array([err], dtype=np.float32), xs_sorted)[0])
                    pathology = int(p_path >= thr)
                    status = "Success"
                except Exception as e:
                    p_path = np.nan
                    pathology = ""
                    status = f"Failure: {type(e).__name__}: {e}"

                rows.append({
                    "path_to_study": path_to_study,
                    "study_uid": suid,
                    "series_uid": srid,
                    "probability_of_pathology": p_path,
                    "pathology": pathology,
                    "processing_status": status,
                    "time_of_processing": time.time() - t0,
                })

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in ("probability_of_pathology", "time_of_processing") else np.nan
    df = df[REQUIRED_COLUMNS]

    def _to_int_or_blank(x):
        try:
            return int(x)
        except Exception:
            return ""
    df["pathology"] = df["pathology"].apply(_to_int_or_blank)

    df.to_excel(args.xlsx_out, index=False)
    print(f"Wrote: {args.xlsx_out}")

if __name__ == "__main__":
    main()
