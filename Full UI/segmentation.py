from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg", force=True)

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from skimage import exposure, feature, filters
from skimage.filters import threshold_otsu
from skimage.morphology import white_tophat, disk, remove_small_objects, binary_closing

from dicom_utils import WorkflowError, sitk_image_to_2d_array


# run the interactive lasso segmentation workflow for one selected hand image
# this saves one mask per bone and returns everything needed for later ratio/curvature steps
def run_lasso_segmentation(dicom_path: Path, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)

    # expected order is fixed so the saved mask names stay consistent across studies
    expected_bone_labels = [
        "distal_phalanx",
        "middle_phalanx",
        "proximal_phalanx",
        "metacarpal",
    ]

    # load the selected DICOM and convert it to a 2D image for display and processing
    img = sitk.ReadImage(str(dicom_path))
    image_2d = sitk_image_to_2d_array(img).astype(np.float32)

    image_min = np.min(image_2d)
    image_max = np.max(image_2d)
    if image_max > image_min:
        image_norm = (image_2d - image_min) / (image_max - image_min)
    else:
        image_norm = np.zeros_like(image_2d, dtype=np.float32)

    # enhance bright bony structures before the user starts drawing
    # this makes thresholding inside the lasso region more stable
    bone_enhanced = white_tophat(image_norm, disk(15))
    bone_enhanced = filters.median(bone_enhanced)
    clahe_image = exposure.equalize_adapthist(bone_enhanced, clip_limit=0.03)

    # build the interactive window that shows instructions and the current overlay
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.canvas.manager.set_window_title("Bone Segmentation")
    plt.subplots_adjust(top=0.65)

    fig.suptitle(
        "Freehand lasso:\n"
        "Hold LEFT mouse and draw, then release.\n\n"
        "Keyboard controls:\n"
        "Enter = Process ROI\n"
        "c = Clear current ROI\n"
        "u = Undo last saved segment\n"
        "q = Finish and close\n"
        " \n",
        fontsize=11,
        y=0.95,
    )

    ax.imshow(clahe_image, cmap="gray")
    ax.set_title(
        "Segment Bones in the Ring Finger in the following order:\n"
        "Order: distal -> middle -> proximal -> metacarpal",
        fontsize=12,
        pad=14,
    )
    ax.axis("off")

    h, w = clahe_image.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])

    overlay = np.zeros((h, w, 4), dtype=float)
    overlay_im = ax.imshow(overlay)

    current_mask = {"value": None}
    processed_masks = []
    saved_paths = []
    assigned_names = []

    drawing_state = {
        "is_drawing": False,
        "verts": [],
    }

    draw_line, = ax.plot([], [], color="red", linewidth=2)

    # update the coloured overlay so the user can see finished masks and the current selection
    def redraw_overlay():
        overlay[:] = 0

        combined = np.zeros((h, w), dtype=bool)
        for mask in processed_masks:
            combined |= mask

        if current_mask["value"] is not None:
            overlay[current_mask["value"]] = [0, 1, 0, 0.35]

        overlay[combined] = [1, 0, 0, 0.25]
        overlay_im.set_data(overlay)

        if len(processed_masks) < len(expected_bone_labels):
            next_label = expected_bone_labels[len(processed_masks)]
        else:
            next_label = "optional extra segment"

        completed = ", ".join(assigned_names) if assigned_names else "none"
        ax.set_xlabel(
            f"Next segment: {next_label} | Completed: {completed}",
            fontsize=10,
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def clear_draw_line():
        draw_line.set_data([], [])
        fig.canvas.draw_idle()

    # convert the hand-drawn polygon into a binary ROI
    # then threshold only inside that ROI so the final mask follows the bone better
    def finalize_polygon(verts):
        if len(verts) < 3:
            print("SEG: not enough points in polygon")
            return

        path = MplPath(verts)
        mask = path.contains_points(pixels).reshape(h, w)
        roi_vals = clahe_image[mask]

        if roi_vals.size == 0:
            print("SEG: ROI has no pixels")
            return

        t = threshold_otsu(roi_vals)

        seg = np.zeros_like(clahe_image, dtype=bool)
        seg[mask] = clahe_image[mask] > t

        current_mask["value"] = seg
        print("SEG: current mask created")
        redraw_overlay()

    # start storing vertices when the left mouse button is pressed on the image
    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        drawing_state["is_drawing"] = True
        drawing_state["verts"] = [(event.xdata, event.ydata)]
        draw_line.set_data([event.xdata], [event.ydata])
        fig.canvas.draw_idle()

    # keep drawing the red line while the mouse is moving
    def on_motion(event):
        if not drawing_state["is_drawing"]:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        drawing_state["verts"].append((event.xdata, event.ydata))
        xs, ys = zip(*drawing_state["verts"])
        draw_line.set_data(xs, ys)
        fig.canvas.draw_idle()

    # close the polygon when the mouse is released and build the current candidate mask
    def on_release(event):
        if not drawing_state["is_drawing"]:
            return

        drawing_state["is_drawing"] = False

        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            drawing_state["verts"].append((event.xdata, event.ydata))

        verts = drawing_state["verts"][:]
        drawing_state["verts"] = []

        clear_draw_line()
        print("SEG: mouse released, finalizing polygon")
        finalize_polygon(verts)

    def process_current_mask():
        if current_mask["value"] is None:
            print("SEG: No current mask to process")
            return

        clean_mask = remove_small_objects(current_mask["value"], min_size=200)
        clean_mask = binary_closing(clean_mask, disk(5))
        edges = feature.canny(image_norm, sigma=1.5)
        enhanced_mask = clean_mask | (clean_mask & edges)

        processed_masks.append(enhanced_mask)
        current_mask["value"] = None

        label_idx = len(processed_masks) - 1
        if label_idx < len(expected_bone_labels):
            segment_label = expected_bone_labels[label_idx]
        else:
            segment_label = f"extra_segment_{label_idx + 1}"

        assigned_names.append(segment_label)

        mask_path = save_dir / f"{segment_label}.png"
        plt.imsave(mask_path, (enhanced_mask * 255).astype(np.uint8), cmap="gray")
        saved_paths.append(str(mask_path))

        print(f"SEG: Saved segment: {segment_label}")
        redraw_overlay()

    def clear_current():
        drawing_state["is_drawing"] = False
        drawing_state["verts"] = []
        current_mask["value"] = None
        clear_draw_line()
        redraw_overlay()
        print("SEG: Cleared current ROI")

    def undo_last():
        if not processed_masks:
            print("SEG: No processed masks to undo")
            return

        processed_masks.pop()
        if assigned_names:
            removed_name = assigned_names.pop()
        else:
            removed_name = "last segment"

        if saved_paths:
            saved_paths.pop()

        redraw_overlay()
        print(f"SEG: Undid segment: {removed_name}")

    def save_sum_mask():
        if not processed_masks:
            print("SEG: No masks yet, sum mask not saved")
            return

        combined = np.zeros((h, w), dtype=bool)
        for mask in processed_masks:
            combined |= mask

        sum_path = save_dir / "sum_mask.png"
        plt.imsave(sum_path, (combined * 255).astype(np.uint8), cmap="gray")
        print("SEG: sum mask saved")

    # keyboard controls
    # Enter = accept current mask
    # c = clear current mask
    # u = undo last saved mask
    # q = finish the segmentation window
    def on_key(event):
        print("SEG: key pressed ->", event.key)
        if event.key in ("enter", "return"):
            process_current_mask()
        elif event.key == "c":
            clear_current()
        elif event.key == "u":
            undo_last()
        elif event.key == "q":
            print("SEG: Q PRESSED - closing segmentation window")
            save_sum_mask()

            plt.close(fig)
            plt.close("all")

            import matplotlib.pyplot as plt_local
            plt_local.pause(0.01)

    # connect mouse and keyboard callbacks to the figure
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw_overlay()

    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

    print("SEG: SEGMENTATION WINDOW CLOSED")
    print("SEG: processed_masks =", len(processed_masks))

    combined = np.zeros((h, w), dtype=bool)
    for mask in processed_masks:
        combined |= mask

    sum_path = save_dir / "sum_mask.png"
    plt.imsave(sum_path, (combined * 255).astype(np.uint8), cmap="gray")
    print("SEG: final sum mask saved")

    return {
        "dicom_path": str(dicom_path),
        "saved_masks": saved_paths,
        "sum_mask": str(sum_path),
        "mask_names": assigned_names,
        "masks": processed_masks,
        "image_array": image_2d,
        "status": "complete",
    }
