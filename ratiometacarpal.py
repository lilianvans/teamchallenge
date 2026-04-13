import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pandas as pd
import matplotlib.pyplot as plt

### change this 

root_folder = r"Segmentations"
output_excel = r"results_all.xlsx"


def ratio(mask):

    # check if mask is 3D and convert to 2D if necessary
    if mask.ndim == 3:
        mask = mask[:, :, 0]  

    # binarize mask
    mask = mask > 0  

    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        raise ValueError("Mask empty.")

    pca = PCA(n_components=2)
    pca.fit(coords)
    long_axis = pca.components_[0]  # main direction
    
    # long axis check
    print("Long axis vector:", long_axis)
    if long_axis[0] < 0:   #always same direction
        
        print("Long axis points left, flipping to right.")
        long_axis = -long_axis

    # project coordinates onto long axis 
    proj = np.dot(coords, long_axis)
    # get length in pixels
    length_pixels = proj.max() - proj.min()

    # band width in pixels 
    band_width = 0.01 * length_pixels 
   
    # perpendicular axis
    perp_axis = np.array([-long_axis[1], long_axis[0]])

    # position mid point of object 
    mid_proj = (proj.max() + proj.min()) / 2

    # position 90% of object
    pos_90 = proj.min() + 0.9 * length_pixels

    # position 10% of object
    pos_10 = proj.min() + 0.1 * length_pixels

    pos=[pos_10, mid_proj, pos_90]
    width=dict()

    for i in pos:
        band = np.abs(proj - i) < band_width  # boolean for band
        coords_band = coords[band]
        width_proj = coords_band @ perp_axis
        width_pixels = width_proj.max() - width_proj.min()
        width[i] = width_pixels
        print(f"Width at position {i:.2f}: {width_pixels:.2f} pixels")

     

    length_mm = length_pixels
    width_mid_mm = width[pos[1]] 
    width_90_mm = width[pos[2]] 
    width_10_mm = width[pos[0]] 

    
    
    print(f"Length: {length_mm:.2f} mm, Width: {width_mid_mm:.2f} mm")
    print(f"Width at 90% position: {width_90_mm:.2f} mm")
    print(f"Width at 10% position: {width_10_mm:.2f} mm")
    ratio = calculate_ratio(length_mm, width_mid_mm)
    print(f"Ratio (length/width): {ratio:.2f}")
    ratio_90 = calculate_ratio(length_mm, width_90_mm)
    print(f"Ratio at 90% position: {ratio_90:.2f}")
    ratio_10 = calculate_ratio(length_mm, width_10_mm)
    print(f"Ratio at 10% position: {ratio_10:.2f}")
    ratio_width= calculate_ratio(width_90_mm, width_10_mm)
    print(f"Ratio width (width_90/width_10): {ratio_width:.2f}")

    return ratio, ratio_90, ratio_10, ratio_width


def calculate_ratio(length, width):
    if width == 0:
        return float('inf')  # Avoid division by zero
    return length / width


def excel_output(file_path, subject, date, series,
                 ratio, ratio_90, ratio_10, ratio_width, bone_type ):

    import pandas as pd
    from pathlib import Path

    # new row of data to add
    new_data = {
        'Subject': subject,
        'Date': date,
        'serial': series,
        'Ratio (length/width)': ratio,
        'Ratio at 90% position': ratio_90,
        'Ratio at 10% position': ratio_10,
        'Ratio width (width_90/width_10)': ratio_width
    }

    new_df = pd.DataFrame([new_data])

    file_path = Path(file_path)

    # Check if file exists and combine with existing data
    if file_path.exists():
        existing_df = pd.read_excel(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # save
    combined_df.to_excel(file_path, index=False)

    print(f"Saved data for {subject} - {bone_type}")



def process_folder(root_folder, output_excel):

    all_rows = []

    for hand_folder in os.listdir(root_folder):
        hand_path = os.path.join(root_folder, hand_folder)

        if not os.path.isdir(hand_path):
            continue

        print(f"\nProcessing: {hand_folder}")

        row_data = {"Hand": hand_folder}

        for file in os.listdir(hand_path):

            if not file.endswith(".png"):
                continue

            if file == "sum_mask.png":
                continue 

            file_path = os.path.join(hand_path, file)

            print(f"  Bone: {file}")

            mask = plt.imread(file_path)

            try:
                r, r90, r10, rw = ratio(mask)
            except Exception as e:
                print(f"  Error in {file}: {e}")
                continue

            # bone naam schoonmaken
            bone_name = os.path.splitext(file)[0].replace(" ", "_")

            # opslaan in kolommen
            row_data[f"{bone_name}_ratio"] = r
            row_data[f"{bone_name}_ratio90"] = r90
            row_data[f"{bone_name}_ratio10"] = r10
            row_data[f"{bone_name}_ratio_width"] = rw

        all_rows.append(row_data)

    # naar dataframe
    df = pd.DataFrame(all_rows)

    # opslaan naar excel
    df.to_excel(output_excel, index=False)

    print(f"\n Excel opgeslagen: {output_excel}")



process_folder(root_folder, output_excel)
