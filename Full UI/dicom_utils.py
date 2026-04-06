from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError


# default base folder used by the workflow to store imported patient studies
BASE_DIR = Path("data/patients")


# custom error used across the UI so workflow problems can be shown cleanly to the user
class WorkflowError(Exception):
    pass


# convert patient names to safe folder names
def sanitize_patient_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown_patient"


# convert study names to safe folder names
def sanitize_study_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown_study"


# build a readable patient label from the DICOM header for the UI
def get_patient_display_from_dicom(ds) -> str:
    patient_name = str(getattr(ds, "PatientName", "")).strip()
    patient_id = str(getattr(ds, "PatientID", "")).strip()

    if patient_name and patient_id:
        return f"{patient_name} ({patient_id})"
    if patient_name:
        return patient_name
    if patient_id:
        return patient_id
    return "Unknown patient"


# choose a stable patient folder name, preferring PatientID over PatientName
def get_patient_folder_name_from_dicom(ds) -> str:
    patient_name = str(getattr(ds, "PatientName", "")).strip()
    patient_id = str(getattr(ds, "PatientID", "")).strip()

    if patient_id:
        return sanitize_patient_name(patient_id)
    if patient_name:
        return sanitize_patient_name(patient_name)
    return "unknown_patient"


def _parse_yyyymmdd(value: str | None):
    value = str(value or "").strip()
    if len(value) >= 8 and value[:8].isdigit():
        try:
            return datetime.strptime(value[:8], "%Y%m%d")
        except ValueError:
            return None
    return None


def _dicom_age_string_to_months(age_str: str | None) -> float | None:
    age_str = str(age_str or "").strip()
    if len(age_str) != 4 or not age_str[:3].isdigit():
        return None

    value = int(age_str[:3])
    unit = age_str[3].upper()

    if unit == "Y":
        return float(value * 12)
    if unit == "M":
        return float(value)
    if unit == "W":
        return float(value) / 4.34524
    if unit == "D":
        return float(value) / 30.4375
    return None


# try several common DICOM date fields and return the first valid one
def get_study_date_from_dicom(ds) -> str:
    for attr in ("StudyDate", "AcquisitionDate", "SeriesDate", "ContentDate"):
        value = str(getattr(ds, attr, "")).strip()
        if len(value) >= 8 and value[:8].isdigit():
            return value[:8]
    return ""


# prefer exact age from birth date and study date
# only fall back to the DICOM PatientAge string if needed
def get_patient_age_months_from_dicom(ds) -> float | None:
        # Prefer the more precise age from PatientBirthDate + study date.
    # Fall back to DICOM PatientAge only if the dates are unavailable.

    birth_dt = _parse_yyyymmdd(getattr(ds, "PatientBirthDate", None))
    study_dt = _parse_yyyymmdd(get_study_date_from_dicom(ds))

    if birth_dt is not None and study_dt is not None:
        delta_days = (study_dt - birth_dt).days
        if delta_days >= 0:
            return round(delta_days / 30.4375, 1)

    patient_age_months = _dicom_age_string_to_months(getattr(ds, "PatientAge", None))
    if patient_age_months is not None:
        return round(patient_age_months, 1)

    return None


def get_patient_age_years_from_dicom(ds) -> float | None:
    months = get_patient_age_months_from_dicom(ds)
    if months is None:
        return None
    return round(months / 12.0, 4)


# placeholder helper
# bone age is not standard in DICOM, so this only checks a few custom field names
def get_bone_age_months_from_dicom(ds) -> float | None:
        # Placeholder helper.
    # Bone age is not consistently available as a standard DICOM field.
    # Extend later if your exported data includes it.

    for attr in ("BoneAgeMonths", "BoneAge", "EstimatedBoneAgeMonths"):
        value = getattr(ds, attr, None)
        if value is None:
            continue
        try:
            return round(float(value), 2)
        except Exception:
            continue
    return None


def get_study_year_from_dicom(ds) -> str:
    for attr in ("StudyDate", "AcquisitionDate", "SeriesDate", "ContentDate"):
        value = str(getattr(ds, attr, "")).strip()
        if len(value) >= 4 and value[:4].isdigit():
            return value[:4]
    return "Unknown"


# build a readable study name for the study list in the UI
def get_study_display_from_dicom(ds, fallback_filename: str = "") -> str:
    year = get_study_year_from_dicom(ds)
    series_desc = str(getattr(ds, "SeriesDescription", "")).strip()
    study_desc = str(getattr(ds, "StudyDescription", "")).strip()
    instance = str(getattr(ds, "InstanceNumber", "")).strip()

    parts = [year]

    if study_desc:
        parts.append(study_desc)
    elif series_desc:
        parts.append(series_desc)

    if instance:
        parts.append(f"Instance {instance}")

    if fallback_filename:
        parts.append(fallback_filename)

    return " | ".join(parts)


def get_study_folder_name_from_dicom(ds, dicom_path: Path | None = None) -> str:
    year = get_study_year_from_dicom(ds)

    study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
    series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
    series_number = str(getattr(ds, "SeriesNumber", "")).strip()

    if study_uid:
        unique_part = sanitize_study_name(study_uid)
    elif series_uid:
        unique_part = sanitize_study_name(series_uid)
    elif dicom_path is not None:
        unique_part = sanitize_study_name(dicom_path.stem)
    elif series_number:
        unique_part = f"series_{sanitize_study_name(series_number)}"
    else:
        unique_part = "unknown_study"

    return f"{year}_{unique_part}"


def ensure_patient_structure(folder_name: str, base_dir: Path = BASE_DIR) -> Path:
        # Create:
    # base_dir / patient_folder
    # / studies
    # / analysis_data

    patient_dir = Path(base_dir) / folder_name
    patient_dir.mkdir(parents=True, exist_ok=True)

    (patient_dir / "studies").mkdir(exist_ok=True)
    (patient_dir / "analysis_data").mkdir(exist_ok=True)

    return patient_dir


def ensure_study_structure(
    patient_folder: str,
    study_folder: str,
    base_dir: Path = BASE_DIR,
) -> Path:
        # Create:
    # base_dir / patient_folder / studies / study_folder
    # / study_files
    # / segmentations
    # / outputs
    # / hip_outputs

    patient_dir = Path(base_dir) / patient_folder
    study_dir = patient_dir / "studies" / study_folder
    study_dir.mkdir(parents=True, exist_ok=True)

    (study_dir / "study_files").mkdir(exist_ok=True)
    (study_dir / "segmentations").mkdir(exist_ok=True)
    (study_dir / "outputs").mkdir(exist_ok=True)
    (study_dir / "hip_outputs").mkdir(exist_ok=True)

    return study_dir


def copy_dicom_to_study(dicom_path: Path, study_dir: Path) -> Path:
        # Copy a selected DICOM into the study's study_files folder.

    dicom_path = Path(dicom_path)
    study_files_dir = Path(study_dir) / "study_files"
    study_files_dir.mkdir(parents=True, exist_ok=True)

    destination = study_files_dir / dicom_path.name

    # avoid accidental overwrite collisions
    if destination.exists():
        stem = dicom_path.stem
        suffix = dicom_path.suffix
        counter = 1
        while True:
            candidate = study_files_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                destination = candidate
                break
            counter += 1

    shutil.copy2(dicom_path, destination)
    return destination


def dicom_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    arr = pixel_array.astype(np.float32)
    arr -= np.min(arr)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    arr *= 255.0
    return arr.astype(np.uint8)


def sitk_image_to_2d_array(img: sitk.Image) -> np.ndarray:
    pixel_array = sitk.GetArrayFromImage(img).astype(float)
    if pixel_array.ndim == 3:
        return pixel_array[0]
    return pixel_array


# load a DICOM image and convert it into an 8-bit preview image for the UI
def load_dicom_for_preview(dicom_path: Path):
        # Returns:
    # ds, preview_array_uint8

    try:
        ds = pydicom.dcmread(str(dicom_path))
    except InvalidDicomError:
        raise WorkflowError("Selected file is not a valid DICOM file.")
    except Exception as e:
        raise WorkflowError(f"Could not read DICOM file: {e}")

    if not hasattr(ds, "pixel_array"):
        raise WorkflowError("This DICOM file has no pixel data.")

    try:
        image = ds.pixel_array
    except Exception as e:
        raise WorkflowError(f"Could not extract pixel data from DICOM: {e}")

    if image.ndim == 3:
        image = image[0]

    image = np.asarray(image)

    # handle MONOCHROME1 inversion
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper().strip()
    if photometric == "MONOCHROME1":
        image = np.max(image) - image

    preview_array = dicom_to_uint8(image)
    return ds, preview_array
