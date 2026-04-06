import numpy as np
from skimage import io, measure
from skimage.morphology import opening, disk
from pathlib import Path
import matplotlib.pyplot as plt
import math
import os
import csv
from datetime import datetime

#path to the folders or files that we need
ROOT_SEGMENTATIONS_FOLDER = r"Segmentations"
RESULTS_CSV_FILE = "finger_curvature_from_png_results.csv"

#add a path to the separate bone masks
BONE_FILES = {
    "Distal": "distal_phalanx.png",
    "Middle": "middle_phalanx.png",
    "Proximal": "proximal_phalanx.png",
    "Metacarpal": "metacarpal.png"
}

#function to find the principal direction of a bone segment
def get_true_long_axis(mask):
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
    
    #covariance matrix to find the direction of highest variance (the long way)
    cov = np.cov(x, y)
    evals, evecs = np.linalg.eig(cov)
    
    sort_indices = np.argsort(evals)[::-1]
    long_axis_vec = evecs[:, sort_indices[0]]
    
    dx, dy = long_axis_vec[0], long_axis_vec[1]

    return (x0, y0), (dx, dy), region.axis_major_length

def get_metadata_from_folder_name(folder_name):
    """
    Gets data from the patient using the folder name structure
    """
    pid = "Unknown_ID"
    formatted_date = "Unknown_Date"
    serial = "Unknown_Serial"
    img_num = "Unknown_Img"

    if "_" in folder_name:
        parts = folder_name.split("_")
        if len(parts) >= 1:
            pid = parts[0]
        if len(parts) >= 2:
            try:
                formatted_date = datetime.strptime(parts[1], "%Y%m%d").strftime("%d-%m-%Y")
            except ValueError:
                formatted_date = parts[1]
        if len(parts) >= 3:
            serial = parts[2]
        if len(parts) >= 4:
            img_num = parts[3]
            
    return pid, formatted_date, serial, img_num


#function that basically does the whole curvature analysis
def analyze_finger_geometry(folder_path, pat_id, scan_dt, serial_num, img_num):
    sum_mask_path = os.path.join(folder_path, "sum_mask.png")
    if not os.path.exists(sum_mask_path):
        print(f"sum_mask.png not found in {folder_path}")
        return
     
    #normalize the dicom pixel data for processing (for PNG masks, simple threshold)
    binary_mask = io.imread(sum_mask_path, as_gray=True) > 0.5
    #perform a subtle opening operation to separate connected segments
    #this version is used for both the calculation AND the background plot
    binary_mask = opening(binary_mask, disk(3))

    #rop the image, removing not needed space
    #find indices of all trues
    coordinates = np.argwhere(binary_mask)

    if coordinates.size == 0:
        return # if empty, return mask

    #find the rectangle space in which all trues lie
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    
    #set amount of pixels for padding
    padding = 20

    #ensure a bit of padding & avoid negative indices & going out of bounds
    y_start = max(0, y_min - padding)
    y_end   = min(binary_mask.shape[0], y_max + padding + 1)
    
    x_start = max(0, x_min - padding)
    x_end   = min(binary_mask.shape[1], x_max + padding + 1)

    cropped_mask = binary_mask[y_start:y_end, x_start:x_end]

    bone_data = {}
    bone_names = ["Distal", "Middle", "Proximal", "Metacarpal"]
    colors = ["#FF2121", "#29FF29", "#1818FF", "#FFFF03"] #red, green, blue, yellow for axes

    #process individual bones using coordinates from the full mask
    for name in bone_names:
        file_path = os.path.join(folder_path, BONE_FILES[name])
        if os.path.exists(file_path):
            mask = io.imread(file_path, as_gray=True)
            axis = get_true_long_axis(mask)
            if axis:
                bone_data[name] = {'centroid': axis[0], 'vec': axis[1], 'len': axis[2]}

    #plotting
    fig, (ax_text, ax_img) = plt.subplots(1, 2, figsize=(8, 10), gridspec_kw={'width_ratios': [1, 2]})
    ax_text.axis('off')
    ax_img.imshow(cropped_mask, cmap='gray', alpha=0.6)
    ax_img.axis('off')

    results_angles = {"DIP": "NaN", "PIP": "NaN", "MCP": "NaN"}

    #start calculation axes and draw segments
    for i, name in enumerate(bone_names):
        if name in bone_data:
            d = bone_data[name]
            global_x, global_y = d['centroid']
            dx, dy = d['vec']
            l = d['len'] * 0.8 #length of the line to draw
            
            #map global coordinates to the cropped positions for plotting
            x0 = global_x - x_start
            y0 = global_y - y_start

        #draw the dashed longitudinal axis for each bone
        ax_img.plot((x0 - dx*l, x0 + dx*l), (y0 - dy*l, y0 + dy*l), 
                color=colors[i], linewidth=1.5, linestyle='--', label=f'{bone_names[i]} axis')
        
        #show the centroid and label the bone
        ax_img.scatter(x0, y0, color='black', s=30, zorder=3)
        ax_img.text(x0 + 25, y0, f"{bone_names[i]}", color=colors[i], fontsize=10)

        #calculation of the different angles
        if i < len(bone_names) - 1:
            next_name = bone_names[i+1]
            if next_name in bone_data:
                v1, v2 = d['vec'], bone_data[next_name]['vec']
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag1, mag2 = math.sqrt(v1[0]**2+v1[1]**2), math.sqrt(v2[0]**2+v2[1]**2)
                angle_diff = math.degrees(math.acos(np.clip(dot/(mag1*mag2), -1.0, 1.0)))
                
                key = "DIP" if name == "Distal" else "PIP" if name == "Middle" else "MCP"
                results_angles[key] = round(angle_diff, 2)
        
    #text display
    total_flexion = round(results_angles["DIP"] + results_angles["PIP"], 2) if (results_angles["DIP"] != "NaN" and results_angles["PIP"] != "NaN") else "NaN"
    #calculate joint flexion using vector dot product
    results_text = "FLEXION ANALYSIS\n" + "\n\n"
    results_text += f"DIP Joint: {results_angles['DIP']}°\n"
    results_text += f"PIP Joint: {results_angles['PIP']}°\n"
    results_text += f"MCP Joint: {results_angles['MCP']}°\n\n"
    results_text += f"\nOverall Finger Flexion (excl. MCP joint): {total_flexion}°"

    ax_text.text(0.1, 0.9, results_text, color='black', fontsize=11, fontweight='bold', verticalalignment='top')
    plt.suptitle(f"{pat_id} | Date: {scan_dt} | Img: {img_num}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #export the results to a csv, including patient characteristics
    file_exists = os.path.isfile(RESULTS_CSV_FILE)
    with open(RESULTS_CSV_FILE, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Patient_ID", "Date", "Serial_Number", "Image_Number", "Filename", "Total_Flexion", "DIP_Angle", "PIP_Angle", "MCP_Angle"])
            
        writer.writerow([pat_id, scan_dt, serial_num, img_num, os.path.basename(folder_path), 
                         total_flexion,results_angles['DIP'], results_angles['PIP'], results_angles['MCP']])

#main loop that combines everything
if __name__ == "__main__":
    if os.path.isdir(ROOT_SEGMENTATIONS_FOLDER):
        #look for the files that we need in our main data folder
        for folder_name in os.listdir(ROOT_SEGMENTATIONS_FOLDER):
            folder_path = os.path.join(ROOT_SEGMENTATIONS_FOLDER, folder_name)
            if os.path.isdir(folder_path):
                pat_id, scan_dt, serial_num, img_num = get_metadata_from_folder_name(folder_name) 
                #run main rool
                analyze_finger_geometry(folder_path, pat_id, scan_dt, serial_num, img_num)
    else:
        print(f"Directory not found: {ROOT_SEGMENTATIONS_FOLDER}")
