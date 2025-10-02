# data_utils.py

import os, time, zipfile, io, glob
import numpy as np
import pydicom
import cv2
from typing import List, Tuple, Dict

def read_dicom_pixels(fp: str):

    ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    return hu, ds

def window_and_normalize(hu: np.ndarray, wl: float = -300.0, ww: float = 1400.0) -> np.ndarray:

    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    x = np.clip(hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-7)
    return x

def list_dicom_files(root_dir: str) -> List[str]:

    files: List[str] = []
    if root_dir is None or (not os.path.isdir(root_dir)):
        return files
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            fp = os.path.join(r, f)
            try:
                _ = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                files.append(fp)
            except Exception:
                continue
    return sorted(files)

def load_single_dicom(fp: str, img_size: int = 256):

    hu, ds = read_dicom_pixels(fp)
    img = window_and_normalize(hu)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32)[..., None]  # [H, W, 1]
    meta = {
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "path": fp,
    }
    return x, meta

def load_study_slices_from_dir(study_dir: str, img_size: int = 256):

    dcm_files = sorted(glob.glob(os.path.join(study_dir, "**", "*"), recursive=True))

    dcm_files = [f for f in dcm_files if os.path.isfile(f)]

    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files in {study_dir}")

    slices = []
    meta = {"study_uid": None, "series_uid": None}

    for fp in dcm_files:
        try:
            hu, ds = read_dicom_pixels(fp)
        except Exception as e:

            print(f"[WARN] Skipping {fp}: {e}")
            continue

        if meta["study_uid"] is None:
            meta["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
        if meta["series_uid"] is None:
            meta["series_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))

        img = window_and_normalize(hu)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        slices.append(img[..., None])

    if len(slices) == 0:
        raise ValueError(f"No valid DICOM slices found in {study_dir}")

    x = np.stack(slices, axis=0).astype(np.float32)
    return x, meta

def iter_study_dirs(root_dir: str):

    if root_dir is None or (not os.path.isdir(root_dir)):
        return []

    subs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if subs:
        return sorted(subs)

    files = [f for f in glob.glob(os.path.join(root_dir, "*")) if os.path.isfile(f)]
    if files:
        return [root_dir]

    return []

def _list_all_files(root_dir: str):
    paths = []
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            paths.append(os.path.join(r, f))
    return paths

def group_dicom_files_by_uid_in_dir(root_dir: str, by_series: bool = False):

    groups: Dict[Tuple[str, str], list] = {}
    files = _list_all_files(root_dir)

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            study_uid = str(getattr(ds, "StudyInstanceUID", "")) or ""
            series_uid = str(getattr(ds, "SeriesInstanceUID", "")) or ""
            if not study_uid:
                continue
            key = (study_uid, series_uid) if by_series else (study_uid, "")
            groups.setdefault(key, []).append(fp)
        except Exception:

            continue

    out = []
    for (suid, srid), flist in groups.items():
        out.append({"files": sorted(flist), "study_uid": suid, "series_uid": srid})

    if not out and files:
        out = [{"files": sorted(files), "study_uid": "", "series_uid": ""}]
    return out

def extract_zip_to_memory(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        return {info.filename: zf.read(info) for info in zf.infolist() if not info.is_dir()}

def group_dicom_files_by_study_from_zip(zip_path: str, temp_dir: str):
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    groups = []
    for root, dirs, files in os.walk(temp_dir):
        file_paths = [os.path.join(root, f) for f in files]
        dicoms = []
        for fp in file_paths:
            try:
                _ = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                dicoms.append(fp)
            except Exception:
                continue
        if dicoms:
            groups.append((root, sorted(dicoms)))
    return groups

def load_study_from_filelist(file_list, img_size: int = 256):

    slices = []
    meta = {"study_uid": None, "series_uid": None}
    for fp in sorted(file_list):
        try:
            hu, ds = read_dicom_pixels(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
            continue
        if meta["study_uid"] is None:
            meta["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
        if meta["series_uid"] is None:
            meta["series_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))
        img = window_and_normalize(hu)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        slices.append(img[..., None])
    if len(slices) == 0:
        raise ValueError("No valid DICOM slices in provided file list.")
    x = np.stack(slices, axis=0).astype(np.float32)
    return x, meta
