import os, io, json, time, zipfile, shutil, tempfile, uuid, logging, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

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

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
DEFAULT_IMG_SIZE = 256
UPLOAD_LIMIT_MB = 2000048

app = Flask(__name__)
app.secret_key = "replace-me"
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_LIMIT_MB * 1024 * 1024

RUNS: dict[str, dict] = {}

def load_artifacts(artifacts_dir: str, img_size_cli: int | None):
    model_path     = os.path.join(artifacts_dir, "model.keras")
    best_full_path = os.path.join(artifacts_dir, "best_model.keras")
    ecdf_path      = os.path.join(artifacts_dir, "calibration_ecdf.npy")
    stats_path     = os.path.join(artifacts_dir, "stats.json")

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    elif os.path.exists(best_full_path):
        model = tf.keras.models.load_model(best_full_path)
    else:
        raise FileNotFoundError("No model.keras / best_model.keras in artifacts dir")

    if not os.path.exists(ecdf_path):
        raise FileNotFoundError("calibration_ecdf.npy not found in artifacts dir")
    xs_sorted = np.load(ecdf_path)

    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

    img_size = img_size_cli if img_size_cli is not None else int(stats.get("img_size", DEFAULT_IMG_SIZE))
    saved_threshold = stats.get("threshold", None)

    return model, xs_sorted, img_size, saved_threshold

def header_checks(fp: str):
    """
    Быстрая проверка заголовка DICOM.
    Требования: Modality=='CT' и BodyPartExamined=='CHEST'
    """
    if pydicom is None:
        raise ImportError("pydicom is required")

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

    ok = len(reasons) == 0
    meta = {"study_uid": study_uid, "series_uid": series_uid}
    return ok, reasons, meta

def analyze_zip(zip_bytes: bytes, artifacts_dir: str, img_size: int | None, threshold_cli: float | None):
    """
    Принимает байты ZIP, возвращает:
      summary: dict
      df:      DataFrame со всеми подробностями
      thr:     порог, по которому классифицировали
    """
    model, xs_sorted, img_size_eff, saved_thr = load_artifacts(artifacts_dir, img_size)
    if threshold_cli is not None:
        thr = float(threshold_cli)
    elif saved_thr is not None:
        thr = float(saved_thr)
    else:
        thr = 0.5

    tmp_root = tempfile.mkdtemp(prefix="flask_ct_")
    extract_dir = os.path.join(tmp_root, "unzipped")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(extract_dir)

    files = []
    for r, _, fs in os.walk(extract_dir):
        for f in fs:
            files.append(os.path.join(r, f))
    files.sort()

    rows = []
    start_all = time.time()

    top_dirs = set()
    reason_counter = Counter()
    normals = 0
    paths = 0
    fails = 0

    for fp in files:
        t0 = time.time()
        inner_path = os.path.relpath(fp, extract_dir)
        top = inner_path.split(os.sep, 1)[0] if os.sep in inner_path else "(root)"
        top_dirs.add(top)

        try:
            ok, reasons, meta_hdr = header_checks(fp)
        except Exception as e:
            rows.append({
                "path_to_study": f"{inner_path}",
                "study_uid": "",
                "series_uid": "",
                "probability_of_pathology": np.nan,
                "pathology": "",
                "processing_status": f"Failure: HeaderReadError: {type(e).__name__}: {e}",
                "time_of_processing": time.time() - t0,
            })
            fails += 1
            reason_counter["HeaderReadError"] += 1
            continue

        if not ok:
            msg = "Failure: " + ", ".join(reasons)
            rows.append({
                "path_to_study": f"{inner_path}",
                "study_uid": meta_hdr["study_uid"],
                "series_uid": meta_hdr["series_uid"],
                "probability_of_pathology": np.nan,
                "pathology": "",
                "processing_status": msg,
                "time_of_processing": time.time() - t0,
            })
            fails += 1
            for rsn in reasons:
                reason_counter[rsn] += 1
            continue

        try:
            x, _ = load_single_dicom(fp, img_size=img_size_eff)
            x_hat = model.predict(x[None, ...], verbose=0)[0]
            err = float(np.mean(np.abs(x - x_hat)))
            p_path = float(1.0 - prob_normal_from_error(np.array([err], dtype=np.float32), xs_sorted)[0])
            pred = int(p_path >= thr)
            rows.append({
                "path_to_study": f"{inner_path}",
                "study_uid": meta_hdr["study_uid"],
                "series_uid": meta_hdr["series_uid"],
                "probability_of_pathology": p_path,
                "pathology": pred,
                "processing_status": "Success",
                "time_of_processing": time.time() - t0,
            })
            if pred == 1:
                paths += 1
            else:
                normals += 1
        except Exception as e:
            rows.append({
                "path_to_study": f"{inner_path}",
                "study_uid": meta_hdr["study_uid"],
                "series_uid": meta_hdr["series_uid"],
                "probability_of_pathology": np.nan,
                "pathology": "",
                "processing_status": f"Failure: {type(e).__name__}: {e}",
                "time_of_processing": time.time() - t0,
            })
            fails += 1
            reason_counter[type(e).__name__] += 1

    df = pd.DataFrame(rows, columns=[
        "path_to_study","study_uid","series_uid",
        "probability_of_pathology","pathology",
        "processing_status","time_of_processing"
    ])
    summary = {
        "folders_analyzed": len(top_dirs),
        "slices_analyzed": len(files),
        "normals": int(normals),
        "pathologies": int(paths),
        "failures": int(fails),
        "failure_reasons": dict(sorted(reason_counter.items(), key=lambda x: -x[1])),
        "elapsed": time.time() - start_all,
    }

    job_id = uuid.uuid4().hex[:10]
    job_dir = os.path.join(tmp_root, job_id)
    os.makedirs(job_dir, exist_ok=True)
    xlsx_path = os.path.join(job_dir, f"results_{job_id}.xlsx")
    df.to_excel(xlsx_path, index=False)

    RUNS[job_id] = {
        "xlsx_path": xlsx_path,
        "summary": summary,
        "threshold": thr,
        "img_size": img_size_eff,
        "root": tmp_root,
    }
    return job_id, summary, thr

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "zip_file" not in request.files:
        flash("Загрузите ZIP-файл.")
        return redirect(url_for("index"))

    file = request.files["zip_file"]
    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".zip"):
        flash("Поддерживается только .zip")
        return redirect(url_for("index"))

    img_size = request.form.get("img_size", type=int)
    thr = request.form.get("threshold", type=float)

    data = file.read()
    try:
        job_id, summary, used_thr = analyze_zip(
            data, ARTIFACTS_DIR, img_size or None, thr
        )
    except Exception as e:
        flash(f"Ошибка анализа: {type(e).__name__}: {e}")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        job_id=job_id,
        summary=summary,
        threshold=used_thr
    )

@app.route("/download/<job_id>")
def download(job_id):
    run = RUNS.get(job_id)
    if not run or not os.path.exists(run["xlsx_path"]):
        flash("Файл недоступен или устарел.")
        return redirect(url_for("index"))
    return send_file(run["xlsx_path"], as_attachment=True, download_name=os.path.basename(run["xlsx_path"]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
