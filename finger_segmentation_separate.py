import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

import os

root = Path(r"Hand PA")

# Build an index of all DICOM images, sorted by patient, study date and image series description.
index = {}
for file in root.rglob("*"):
    if not file.is_file():
        # Skip everything that is not a file.
        continue

    # Try to read the DICOM file and skip it if it failed.
    try:
        ds = pydicom.dcmread(file, stop_before_pixels=True)
    except Exception:
        continue

    # Read the patient ID, study date and image series description.
    patient_id = ds.get("PatientID")
    study_date = ds.get("StudyDate")
    series_description = ds.get("SeriesDescription")
    
    # Create new index levels.
    if patient_id not in index:
        index[patient_id] = {}
    if study_date not in index[patient_id]:
        index[patient_id][study_date] = {}
    if series_description not in index[patient_id][study_date]:
        index[patient_id][study_date][series_description] = []


    # Add the file to the index.
    index[patient_id][study_date][series_description].append(file)
        
# Read one of the images.
reader = sitk.ImageSeriesReader()

subject = 'H05' # CHANGE
date = '20201001' # CHANGE
projection = 'Hand PA'

reader.SetFileNames(index[subject][date][projection])
img = reader.Execute()

# Show the image.
img_array = sitk.GetArrayFromImage(img)
plt.imshow(img_array[0], cmap='grey')

def plot(string, image):
    plt.imshow(image, cmap = "gray")
    plt.axis("off")
    plt.title(string)
    plt.show()

# Point selection ------------------------------------------------------------------------------
import skimage
import SimpleITK as sitk
from skimage.filters import threshold_otsu
import numpy as np
from skimage import exposure
from skimage.morphology import white_tophat, disk, remove_small_objects, binary_closing
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path

# Preprocessing (verhoog contrast tussen bot en ander weefsel)
image = img_array[0]
bone_enhanced = white_tophat(image, disk(15))
bone_enhanced = skimage.filters.median(bone_enhanced)
clahe_image = exposure.equalize_adapthist(image, clip_limit=0.03) # Contrast-enhanced image

# Set up for lasso selection operator
fig, ax = plt.subplots()
ax.imshow(clahe_image, cmap="gray")

h, w = clahe_image.shape

    # pixel coordinate grid
yy, xx = np.mgrid[:h, :w]
pixels = np.column_stack([xx.ravel(), yy.ravel()])

    # transparent overlay
overlay = np.zeros((h, w, 4))
overlay_im = ax.imshow(overlay)

    # create list of masks
current_mask = None
current_sum_mask = None
def onselect(verts):
    global current_mask
    path = Path(verts)
    
    mask = path.contains_points(pixels).reshape(h, w)
        
    roi_vals = clahe_image[mask]
        
    t = threshold_otsu(roi_vals)
        
    seg = np.zeros_like(clahe_image, dtype=bool)
    seg[mask] = clahe_image[mask] > t
    
    local_mask = np.zeros_like(clahe_image, dtype=bool)
    local_mask[seg] = True
    current_mask = local_mask
        
    overlay[:] = 0
    overlay[seg] = [0,1,0,0.4]
    
    overlay_im.set_data(overlay)
    fig.canvas.draw_idle()

lasso = LassoSelector(ax, onselect)
base_save_dir = "/Users/petereijgenraam/Documents/Medical Imaging/2025-2026/Team Challenge/Data/Segmentations" # CHANGE
study_folder = f"{subject}_{date}"
save_dir = os.path.join(base_save_dir, study_folder)
os.makedirs(save_dir, exist_ok=True)
processed_masks = []
# Button to finalize segmentation
def process_callback(event):
    global current_mask
    global current_sum_mask
    global processed_masks
    
    if current_mask is None:
        return
    
    clean_mask = remove_small_objects(current_mask, min_size=200)
    clean_mask = binary_closing(clean_mask, disk(5))

    
    #for m in masks:
     #   clean_mask = remove_small_objects(m, min_size=200)
      #  clean_mask = binary_closing(clean_mask, disk(5))
        
    edges = skimage.feature.canny(image, sigma=1.5, low_threshold=0.2, high_threshold=0.5)
    edges_on_mask = clean_mask & edges
    enhanced_mask = clean_mask | edges_on_mask
        
    processed_masks.append(enhanced_mask)
    
    sum_mask = np.zeros_like(clahe_image, dtype=bool)
    for m in processed_masks:
        sum_mask = sum_mask | m
    current_sum_mask = sum_mask

    
    for i, m in enumerate(processed_masks):

        filename = os.path.join(
            save_dir,
            f"phalanx_{i+1}.png"
            )

        plt.imsave(
            filename,
            (m * 255).astype(np.uint8),
            cmap="gray"
            )

    # Show results
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].imshow(clean_mask, cmap="gray")
    ax[0].set_title("Original Segmentation")
    ax[0].axis("off")
    
    ax[1].imshow(edges_on_mask, cmap="gray")
    ax[1].set_title("Edges inside Segmentation")
    ax[1].axis("off")
    
    ax[2].imshow(enhanced_mask, cmap="gray")
    ax[2].set_title("Segmentation + Edges")
    ax[2].axis("off")
    
    plt.show()

def save_sum_callback(event):

    global current_sum_mask

    if current_sum_mask is None:
        return

    filename = os.path.join(save_dir, "sum_mask.png")

    plt.imsave(
        filename,
        (current_sum_mask * 255).astype(np.uint8),
        cmap="gray"
    )
    
# Add button
ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
button = Button(ax_button, "Process")
button.on_clicked(process_callback)

ax_button_sum = plt.axes([0.65, 0.01, 0.1, 0.05])
button_sum = Button(ax_button_sum, "Save Sum")
button_sum.on_clicked(save_sum_callback)

plt.show()
