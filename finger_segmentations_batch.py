import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import os
import skimage
from skimage.filters import threshold_otsu
import numpy as np
from skimage import exposure
from skimage.morphology import white_tophat, disk, remove_small_objects, binary_closing
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path as MPath
from datetime import datetime

root = Path(r"Hand PA")
base_save_dir = r"Segmentations" #folder does not need to exist yet

# HARDCODE LESS: define your range of patients  here, H should be included
START_PID = 'H40'
END_PID = 'H47'

#list of bones for instructions sequence
BONE_NAMES = ["Metacarpal", "Proximal Phalanx", "Middle Phalanx", "Distal Phalanx"]

# Build an index of all DICOM images, sorted by patient, study date, serial and image series description.
index = {}
for file in root.rglob("*.dcm"):
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
    
    # Logic to get the unique serial folder name (e.g., H42_20241101_21776)
    # and the specific image subfolder (e.g., 1, 2, or 3)
    parent_1 = os.path.dirname(file)           # DICOM folder
    img_subfolder = os.path.basename(os.path.dirname(parent_1)) # 1, 2, or 3 folder
    serial_folder = os.path.basename(os.path.dirname(os.path.dirname(parent_1))) # H42_20241101_21776 folder

    # Create new index levels. We add img_subfolder to the key to prevent overwriting
    if patient_id not in index:
        index[patient_id] = {}
    if study_date not in index[patient_id]:
        index[patient_id][study_date] = {}
    if serial_folder not in index[patient_id][study_date]:
        index[patient_id][study_date][serial_folder] = {}
    
    # Create a unique key for this specific image instance (e.g., "1", "2")
    if img_subfolder not in index[patient_id][study_date][serial_folder]:
        index[patient_id][study_date][serial_folder][img_subfolder] = {}
        
    if series_description not in index[patient_id][study_date][serial_folder][img_subfolder]:
        index[patient_id][study_date][serial_folder][img_subfolder][series_description] = []

    # Add the file to the index.
    index[patient_id][study_date][serial_folder][img_subfolder][series_description].append(file)

#filter patients by range
sorted_pids = sorted(index.keys())
try:
    target_pids = sorted_pids[sorted_pids.index(START_PID) : sorted_pids.index(END_PID) + 1]
except ValueError:
    print(f"Start or end patient ID not found")
    target_pids = []

class SegmentationSession:
    #updated init to handle specific folder names and image numbering
    def __init__(self, subject, date, projection, specific_files, img_num, folder_name):
        self.subject = subject
        self.date = date
        self.img_num = img_num
        self.folder_name = folder_name # The original DICOM folder name
        self.bone_index = 0
        self.processed_masks = []
        
        # Read the specific image passed from the loop.
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in specific_files])
        img = reader.Execute()

        # Show the image.
        self.img_array = sitk.GetArrayFromImage(img)
        self.image = self.img_array[0]

        # Preprocessing (verhoog contrast tussen bot en ander weefsel)
        # bone_enhanced = white_tophat(self.image, disk(15)) # Kept from your original logic
        # bone_enhanced = skimage.filters.median(bone_enhanced)
        self.clahe_image = exposure.equalize_adapthist(self.image, clip_limit=0.03) # Contrast-enhanced image

        # Set up for lasso selection operator
        self.fig, self.ax = plt.subplots(figsize=(5, 7))
        self.ax.imshow(self.clahe_image, cmap="gray")

        self.h, self.w = self.clahe_image.shape

        # pixel coordinate grid
        yy, xx = np.mgrid[:self.h, :self.w]
        self.pixels = np.column_stack([xx.ravel(), yy.ravel()])

        # transparent overlay
        self.overlay = np.zeros((self.h, self.w, 4))
        self.overlay_im = self.ax.imshow(self.overlay)

        # create list of masks
        self.current_mask = None
        
        #initialize user interface
        self.update_ui_text()

        self.lasso = LassoSelector(self.ax, self.onselect)
        
        # Add button (Button to finalize segmentation)
        ax_button = plt.axes([0.75, 0.01, 0.15, 0.05])
        self.button = Button(ax_button, "Process")
        self.button.on_clicked(self.process_callback)

        plt.show()

    def update_ui_text(self):
        bone = BONE_NAMES[self.bone_index]
        #folder name in plot as well
        self.ax.set_title(f"Folder: {self.folder_name}\nIMG: {self.img_num} | Target: {bone}", 
                          fontsize=9, fontweight='bold')
        self.fig.canvas.draw_idle()

    def onselect(self, verts):
        #logic from original onselect function by Peter
        path = MPath(verts)
        mask = path.contains_points(self.pixels).reshape(self.h, self.w)
        roi_vals = self.clahe_image[mask]
        
        #make sure no crash if nothing is selected
        if len(roi_vals) == 0: return

        t = threshold_otsu(roi_vals)
            
        seg = np.zeros_like(self.clahe_image, dtype=bool)
        seg[mask] = self.clahe_image[mask] > t
        
        self.current_mask = seg
            
        self.overlay[:] = 0
        self.overlay[seg] = [0,1,0,0.4]
        
        self.overlay_im.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def process_callback(self, event):
        #logic from original process_callback function by peter
        if self.current_mask is None:
            return
        
        clean_mask = remove_small_objects(self.current_mask, min_size=200)
        clean_mask = binary_closing(clean_mask, disk(6))

        #for m in masks:
        #   clean_mask = remove_small_objects(m, min_size=200)
        #  clean_mask = binary_closing(clean_mask, disk(5))

        edges = skimage.feature.canny(self.image, sigma=1.5, low_threshold=0.2, high_threshold=0.5)
        edges_on_mask = clean_mask & edges
        enhanced_mask = clean_mask | edges_on_mask
            
        self.processed_masks.append(enhanced_mask)
        
        #move to next bone or finish
        self.bone_index += 1
        self.overlay[:] = 0
        self.overlay_im.set_data(self.overlay)
        self.current_mask = None

        if self.bone_index < len(BONE_NAMES):
            self.update_ui_text()
        else:
            self.auto_save_and_exit()

    def auto_save_and_exit(self):
        # export the results automatically using the requested folder name format: 
        # patientID_date_serialnumber_imagenumber

        #uncomment this in case of intra/inter-observer analysis 
        #timestamp = datetime.now().strftime("%H%M%S")
        #save_folder_name = f"{self.folder_name}_{self.img_num}_run_{timestamp}"

        #comment this in case of intra/inter-observer analysis
        save_folder_name = f"{self.folder_name}_{self.img_num}"
            
        save_dir = os.path.join(base_save_dir, save_folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        current_sum_mask = np.zeros_like(self.clahe_image, dtype=bool)

        for i, m in enumerate(self.processed_masks):
            bone_filename = BONE_NAMES[i].replace(" ", "_").lower()
            filename = os.path.join(save_dir, f"{bone_filename}.png")

            plt.imsave(filename, (m * 255).astype(np.uint8), cmap="gray")
            current_sum_mask = current_sum_mask | m

        #save sum Mask
        sum_filename = os.path.join(save_dir, "sum_mask.png")
        plt.imsave(sum_filename, (current_sum_mask * 255).astype(np.uint8), cmap="gray")
        
        print(f"Saved all masks to folder: {save_folder_name}")
        plt.close(self.fig)

#loop that can handle multiple patients without hardcoding
for pid in target_pids:
    if pid in index:
        for study_date in sorted(index[pid].keys()):
            for serial_folder in sorted(index[pid][study_date].keys()):
                # Now we loop through each specific subfolder (1, 2, 3...)
                for sub_folder in sorted(index[pid][study_date][serial_folder].keys()):
                    if "Hand PA" in index[pid][study_date][serial_folder][sub_folder]:
                        file_list = index[pid][study_date][serial_folder][sub_folder]["Hand PA"]
                        
                        if file_list:
                            dicom_file = file_list[0]
                            print(f"Starting Patient: {pid} | Folder: {serial_folder} | Image: {sub_folder}")
                            SegmentationSession(pid, study_date, "Hand PA", [dicom_file], sub_folder, serial_folder)

print("Processing finished for the selected range.")
