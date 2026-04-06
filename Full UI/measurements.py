import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from sklearn.decomposition import PCA
from skimage import io, measure

from dicom_utils import WorkflowError, sitk_image_to_2d_array


# -------------------------
# UTILITIES
# -------------------------
def load_masks_from_saved_paths(segmentation_data: dict):
    # if masks are already in memory, use those directly
    if "masks" in segmentation_data and segmentation_data["masks"]:
        return segmentation_data["masks"]

    # otherwise load the saved mask files from disk
    saved_paths = segmentation_data.get("saved_masks", [])
    if not saved_paths:
        raise WorkflowError("No saved masks found.")

    masks = []
    for path in saved_paths:
        mask_img = io.imread(path)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        masks.append(mask_img > 0)

    # keep them in the segmentation dict so later steps can reuse them
    segmentation_data["masks"] = masks
    return masks


def normalize_mask_for_ratio(mask: np.ndarray):
    return mask[np.newaxis, :, :]


def load_dicom_image(dicom_path: Path):
    # load image and convert to a single 2D array
    img = sitk.ReadImage(str(dicom_path))
    arr = sitk_image_to_2d_array(img).astype(np.float32)

    # normalize to 0-1 so previews are easier to show
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    return img, arr


def get_true_long_axis(mask):
    # binarize and find the largest connected component
    binary = mask > 0.5
    labels = measure.label(binary)
    regions = measure.regionprops(labels)

    if not regions:
        return None

    region = max(regions, key=lambda r: r.area)

    y0, x0 = region.centroid
    coords = region.coords

    y = coords[:, 0] - y0
    x = coords[:, 1] - x0

    # covariance matrix to find the direction of highest variance
    cov = np.cov(x, y)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    long_axis_vec = evecs[:, sort_indices[0]]

    dx, dy = long_axis_vec[0], long_axis_vec[1]

    return (x0, y0), (dx, dy), region.axis_major_length


def _fit_pca_axis(mask: np.ndarray):
        # Fit PCA axis to the mask and define projections relative to the bone centroid.

    # Returns:
    # coords, centroid, long_axis, perp_axis, proj_long, p_min, p_max

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = mask > 0
    coords = np.column_stack(np.where(mask))  # (row, col) = (y, x)

    if coords.shape[0] < 5:
        return None

    pca = PCA(n_components=2)
    pca.fit(coords)

    long_axis = pca.components_[0].astype(float)

    # keep direction consistent so visuals stay stable between runs
    if long_axis[0] < 0:
        long_axis = -long_axis

    long_axis = long_axis / np.linalg.norm(long_axis)
    perp_axis = np.array([-long_axis[1], long_axis[0]], dtype=float)

    centroid = coords.mean(axis=0)

    # project relative to the centroid, not the image origin
    proj_long = (coords - centroid) @ long_axis
    p_min = float(proj_long.min())
    p_max = float(proj_long.max())

    if np.isclose(p_min, p_max):
        return None

    return coords, centroid, long_axis, perp_axis, proj_long, p_min, p_max


def _measure_width_and_line(mask: np.ndarray, long_axis: np.ndarray, fraction: float):
        # Measure width at a chosen fraction along the long axis and return
    # the endpoints of the corresponding width line.

    # fraction:
    # 0.10 -> near one end
    # 0.50 -> middle
    # 0.90 -> near the other end

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = mask > 0
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 5:
        return None, None, None, None

    centroid = coords.mean(axis=0)

    proj_long = (coords - centroid) @ long_axis
    p_min = proj_long.min()
    p_max = proj_long.max()

    if np.isclose(p_min, p_max):
        return None, None, None, None

    # choose where along the bone we want to measure
    target_proj = p_min + fraction * (p_max - p_min)

    # perpendicular direction to the bone axis
    perp_axis = np.array([-long_axis[1], long_axis[0]], dtype=float)

    # take a thin slice around the chosen position
    slice_half_thickness = max(1.5, 0.02 * (p_max - p_min))
    keep = np.abs(proj_long - target_proj) <= slice_half_thickness
    slice_coords = coords[keep]

    if slice_coords.shape[0] < 2:
        return None, None, None, None

    # project the slice onto the perpendicular axis to get the width
    proj_perp = (slice_coords - centroid) @ perp_axis
    width_pixels = float(proj_perp.max() - proj_perp.min())

    perp_min = float(proj_perp.min())
    perp_max = float(proj_perp.max())

    axis_point = centroid + target_proj * long_axis

    pt1 = axis_point + perp_min * perp_axis
    pt2 = axis_point + perp_max * perp_axis

    return width_pixels, pt1, pt2, axis_point


# -------------------------
# RATIO VISUAL
# -------------------------
def save_ratio_visual(
    patient_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    segment_name: str,
    length_mm,
    width_mid_mm,
    width_10_mm,
    width_90_mm,
):
    output_dir = patient_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = segment_name.replace(" ", "_").lower()
    output_path = output_dir / f"ratio_visual_{safe_name}.png"

    fitted = _fit_pca_axis(mask)
    if fitted is None:
        return None

    coords, centroid, long_axis, perp_axis, proj_long, p_min, p_max = fitted

    # full long axis endpoints, centred on the bone
    start = centroid + p_min * long_axis
    end = centroid + p_max * long_axis

    # width lines at 10%, middle, and 90%
    _, w10_pt1, w10_pt2, _ = _measure_width_and_line(mask, long_axis, 0.10)
    _, w50_pt1, w50_pt2, _ = _measure_width_and_line(mask, long_axis, 0.50)
    _, w90_pt1, w90_pt2, _ = _measure_width_and_line(mask, long_axis, 0.90)

    fig, (ax_text, ax_img) = plt.subplots(
        1, 2, figsize=(9, 10), gridspec_kw={"width_ratios": [1, 2]}
    )

    ax_text.axis("off")
    ax_text.text(
        0.08,
        0.92,
        f"{segment_name}\n\n"
        f"Length: {length_mm:.2f} mm\n"
        f"Width 10%: {width_10_mm:.2f} mm\n"
        f"Width Mid: {width_mid_mm:.2f} mm\n"
        f"Width 90%: {width_90_mm:.2f} mm\n\n"
        f"Blue: long axis\n"
        f"Green: width 10%\n"
        f"Yellow: width mid\n"
        f"Magenta: width 90%",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    ax_img.imshow(image, cmap="gray")
    ax_img.imshow(np.ma.masked_where(~(mask > 0), mask > 0), alpha=0.28, cmap="Reds")

    # long axis
    ax_img.plot(
        [float(start[1]), float(end[1])],
        [float(start[0]), float(end[0])],
        linestyle="-",
        linewidth=2.5,
        color="deepskyblue",
    )

    def plot_line(ax, pt1, pt2, color):
        if pt1 is None or pt2 is None:
            return
        ax.plot(
            [float(pt1[1]), float(pt2[1])],
            [float(pt1[0]), float(pt2[0])],
            linewidth=2.2,
            color=color,
        )

    plot_line(ax_img, w10_pt1, w10_pt2, "lime")
    plot_line(ax_img, w50_pt1, w50_pt2, "yellow")
    plot_line(ax_img, w90_pt1, w90_pt2, "magenta")

    ax_img.axis("off")
    ax_img.set_title(segment_name)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# -------------------------
# RATIO UTILITIES
# -------------------------
def _measure_width_at_fraction(
    mask: np.ndarray,
    long_axis: np.ndarray,
    fraction: float,
) -> float | None:
    width_pixels, _, _, _ = _measure_width_and_line(mask, long_axis, fraction)
    return width_pixels


# -------------------------
# RATIO CALCULATION
# -------------------------
def calculate_finger_ratios(segmentation_data: dict):
    dicom_path = Path(segmentation_data["dicom_path"])
    patient_dir = dicom_path.parent.parent

    img, image = load_dicom_image(dicom_path)
    masks = load_masks_from_saved_paths(segmentation_data)

    results = []

    # use the in-plane spacing to convert pixels to mm
    spacing = img.GetSpacing()
    pixel_size = float(np.mean(spacing[:2]))

    for idx, mask in enumerate(masks):
        fitted = _fit_pca_axis(mask)
        if fitted is None:
            continue

        coords, centroid, long_axis, perp_axis, proj_long, p_min, p_max = fitted

        # full bone length along the fitted long axis
        length_pixels = float(p_max - p_min)
        length_mm = float(length_pixels * pixel_size)

        # widths at the same positions used in the original scripts
        width_10_px = _measure_width_at_fraction(mask, long_axis, 0.10)
        width_mid_px = _measure_width_at_fraction(mask, long_axis, 0.50)
        width_90_px = _measure_width_at_fraction(mask, long_axis, 0.90)

        width_10_mm = None if width_10_px is None else float(width_10_px * pixel_size)
        width_mid_mm = None if width_mid_px is None else float(width_mid_px * pixel_size)
        width_90_mm = None if width_90_px is None else float(width_90_px * pixel_size)

        if idx < len(segmentation_data.get("mask_names", [])):
            segment_name = segmentation_data["mask_names"][idx]
        else:
            segment_name = f"Segment_{idx + 1}"

        visual_path = save_ratio_visual(
            patient_dir=patient_dir,
            image=image,
            mask=mask,
            segment_name=segment_name,
            length_mm=length_mm,
            width_mid_mm=width_mid_mm if width_mid_mm is not None else 0.0,
            width_10_mm=width_10_mm if width_10_mm is not None else 0.0,
            width_90_mm=width_90_mm if width_90_mm is not None else 0.0,
        )

        results.append(
            {
                "segment_name": segment_name,
                "length_mm": round(length_mm, 2),
                "width_mid_mm": round(width_mid_mm, 2) if width_mid_mm is not None else None,
                "width_10_mm": round(width_10_mm, 2) if width_10_mm is not None else None,
                "width_90_mm": round(width_90_mm, 2) if width_90_mm is not None else None,
                "visual_path": str(visual_path) if visual_path else None,
            }
        )

    return results


# -------------------------
# CURVATURE VISUAL
# -------------------------
def save_curvature_visual(
    patient_dir: Path,
    masks: list[np.ndarray],
    bone_data: dict,
    dip,
    pip,
    mcp,
    total,
):
    output_dir = patient_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "curvature_visual.png"

    if not masks:
        return None

    combined = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined |= (mask > 0)

    fig, (ax_text, ax_img) = plt.subplots(
        1, 2, figsize=(8, 10), gridspec_kw={"width_ratios": [1, 2]}
    )

    ax_text.axis("off")
    ax_text.text(
        0.08,
        0.92,
        "Curvature\n\n"
        f"DIP: {dip}\n"
        f"PIP: {pip}\n"
        f"MCP: {mcp}\n"
        f"Total: {total}",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    ax_img.imshow(np.zeros_like(combined), cmap="gray", vmin=0, vmax=1)
    ax_img.imshow(np.ma.masked_where(~combined, combined), alpha=0.35, cmap="Reds")

    colors = ["cyan", "yellow", "lime", "magenta"]
    ordered_names = ["Distal", "Middle", "Proximal", "Metacarpal"]

    for i, name in enumerate(ordered_names):
        if name not in bone_data:
            continue

        x0, y0 = bone_data[name]["centroid"]
        dx, dy = bone_data[name]["vec"]
        axis_len = float(bone_data[name]["len"]) * 0.45

        x1 = x0 - dx * axis_len
        y1 = y0 - dy * axis_len
        x2 = x0 + dx * axis_len
        y2 = y0 + dy * axis_len

        # draw the long axis for each detected bone
        ax_img.plot([x1, x2], [y1, y2], color=colors[i], linewidth=2.0)
        ax_img.scatter(x0, y0, color="black", s=30, zorder=3)
        ax_img.text(x0 + 25, y0, name, color=colors[i], fontsize=10)

    ax_img.axis("off")
    ax_img.set_title("Curvature")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# -------------------------
# CURVATURE CALCULATION
# -------------------------
def calculate_curvature(segmentation_data: dict):
    masks = load_masks_from_saved_paths(segmentation_data)
    dicom_path = Path(segmentation_data["dicom_path"])
    patient_dir = dicom_path.parent.parent

    mask_names = segmentation_data.get("mask_names", [])

    bone_name_map = {
        "distal": "Distal",
        "middle": "Middle",
        "proximal": "Proximal",
        "metacarpal": "Metacarpal",
    }

    bone_data = {}

    # map each saved mask to the expected bone name
    for idx, mask in enumerate(masks):
        raw_name = mask_names[idx] if idx < len(mask_names) else f"Mask_{idx}"
        lower_name = raw_name.lower()

        mapped_name = None
        for key, pretty_name in bone_name_map.items():
            if key in lower_name:
                mapped_name = pretty_name
                break

        if mapped_name is None:
            continue

        axis = get_true_long_axis(mask)
        if axis is None:
            continue

        bone_data[mapped_name] = {
            "centroid": axis[0],
            "vec": axis[1],
            "len": axis[2],
        }

    bone_order = ["Distal", "Middle", "Proximal", "Metacarpal"]

    results_angles = {
        "DIP": None,
        "PIP": None,
        "MCP": None,
    }

    # compare each neighbouring bone pair with a vector dot product
    for i, name in enumerate(bone_order[:-1]):
        next_name = bone_order[i + 1]

        if name not in bone_data or next_name not in bone_data:
            continue

        v1 = bone_data[name]["vec"]
        v2 = bone_data[next_name]["vec"]

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            continue

        angle_diff = math.degrees(
            math.acos(np.clip(dot / (mag1 * mag2), -1.0, 1.0))
        )

        if name == "Distal":
            results_angles["DIP"] = round(angle_diff, 2)
        elif name == "Middle":
            results_angles["PIP"] = round(angle_diff, 2)
        elif name == "Proximal":
            results_angles["MCP"] = round(angle_diff, 2)

    dip = results_angles["DIP"]
    pip = results_angles["PIP"]
    mcp = results_angles["MCP"]

    # keep the same total definition as before: DIP + PIP
    total = None
    if dip is not None and pip is not None:
        total = round(dip + pip, 2)

    visual_path = save_curvature_visual(
        patient_dir=patient_dir,
        masks=masks,
        bone_data=bone_data,
        dip=dip,
        pip=pip,
        mcp=mcp,
        total=total,
    )

    return {
        "DIP": dip,
        "PIP": pip,
        "MCP": mcp,
        "total": total,
        "visual_path": str(visual_path) if visual_path else None,
    }
