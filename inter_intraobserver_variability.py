import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pingouin as pg

def dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    return 2 * intersection / (mask1.sum() + mask2.sum())

folder_path_hip = "Hip"

num_observers = 4
observ_list = ['A', 'B', 'C', 'D']
num_measurements = 3
measr_list = ['1', '2', '3']
num_samples = 5
samp_list = ['a', 'b', 'c', 'd', 'e']
img_size = 256
columns = ["Ratio_Full", "Angle", "Ratio_L", "Ratio_R"]

hip_data = np.zeros((num_observers, len(columns), num_samples, num_measurements))

filenames_hip = os.listdir(folder_path_hip)


targets_hip = []
raters_hip = []
ratings_hip = []
methods_hip = []
observers_hip = []

for i in range(len(filenames_hip)):
    filename = filenames_hip[i]
    if filename.endswith(".csv"):
        filepath = os.path.join(folder_path_hip, filename)

        # read CSV
        df = pd.read_csv(filepath)

        for j in range(len(columns)):
            column = df[columns[j]].to_numpy()

            hip_data[i, j] = column.reshape(num_samples, num_measurements)
            l = 0
            while l < num_measurements:
                m = 0
                while m < num_samples:
                    idx = l*num_samples + m
                    targets_hip.append(measr_list[l])
                    raters_hip.append(samp_list[m])
                    ratings_hip.append(column[idx])
                    methods_hip.append(columns[j])
                    observers_hip.append(observ_list[i])
                    m += 1
                l += 1


df_hip = pd.DataFrame({"Measurements":targets_hip,
                       "Samples":raters_hip,
                       "Ratings":ratings_hip,
                       "Methods":methods_hip,
                       "Observer":observers_hip})

print(df_hip)


folder_path_finger = "Finger"

folders_finger = os.listdir(folder_path_finger)

finger_data = np.zeros((num_observers, num_samples, num_measurements, img_size, img_size))
img_store = []

dice_finger_intra = []
measr_finger_intra = []
obs_finger_intra =[]
samp_finger_intra = []

dice_finger_inter = []
measr_finger_inter = []
obs_finger_inter =[]
samp_finger_inter = []

files_list_list = []

for i in range(len(folders_finger)):
    folder_observer = folders_finger[i]
    folder_name = Path(folder_path_finger + "/" + folder_observer)
    list_files = []
    for file in folder_name.rglob("sum_mask.png"):
        list_files.append(file)
    files_list_list.append(list_files)

    for j in range(num_samples):
        
        for k in range(num_measurements):
            for l in range(k+1, num_measurements):
                idx_1 = num_measurements*j+k
                img_file_1 = list_files[idx_1]

                img_1 = cv2.imread(img_file_1, cv2.IMREAD_GRAYSCALE)
                # img_1 = cv2.resize(img_1, (img_size, img_size))
                # finger_data[i, j, k] = img_1

                idx_2 = num_measurements*j+l
                img_file_2 = list_files[idx_2]

                img_2 = cv2.imread(img_file_2, cv2.IMREAD_GRAYSCALE)
                # img_2 = cv2.resize(img_2, (img_size, img_size))

                dice = dice_score(img_1, img_2)
                dice_finger_intra.append(dice)
                measr_finger_intra.append(measr_list[k]+measr_list[l])
                samp_finger_intra.append(samp_list[j])
                obs_finger_intra.append(observ_list[i])
    
for i in range(num_observers):
    for l in range(i+1, num_observers):
        for j in range(num_samples):
            for k in range(num_measurements):
                idx = num_measurements*j+k
                img_file_1 = files_list_list[i][idx]
                img_1 = cv2.imread(img_file_1, cv2.IMREAD_GRAYSCALE)

                img_file_2 = files_list_list[l][idx]
                img_2 = cv2.imread(img_file_2, cv2.IMREAD_GRAYSCALE)
                dice = dice_score(img_1, img_2)
                dice_finger_inter.append(dice)
                measr_finger_inter.append(measr_list[k])
                samp_finger_inter.append(samp_list[j])
                obs_finger_inter.append(observ_list[i]+observ_list[l])





        

df_finger_intra = pd.DataFrame({"Measurements":measr_finger_intra,
                       "Samples":samp_finger_intra,
                       "Ratings":dice_finger_intra,
                       "Observer":obs_finger_intra})

df_finger_inter = pd.DataFrame({"Measurements":measr_finger_inter,
                       "Samples":samp_finger_inter,
                       "Ratings":dice_finger_inter,
                       "Observer":obs_finger_inter})

print(df_finger_intra)
# print(df_finger_inter)





# Interobserver variability (Hand)
hand_inter_results = []

for m in ['1', '2', '3']:
    df_temp = df_finger_inter[df_finger_inter["Measurements"] == m]
    icc = pg.intraclass_corr(
        data=df_temp,
        targets='Samples',
        raters='Observer',
        ratings='Ratings'
    )
    hand_inter_results.append(icc.assign(Measurement=m))

hand_inter_results = pd.concat(hand_inter_results)
# print(hand_inter_results)

hand_intra_results = []

for obs in ['A', 'B', 'C', 'D']:
    df_temp = df_finger_intra[df_finger_intra["Observer"] == obs]
    icc = pg.intraclass_corr(
        data=df_temp,
        targets='Samples',
        raters='Measurements',
        ratings='Ratings'
    )
    hand_intra_results.append(icc.assign(Observer=obs))

hand_intra_results = pd.concat(hand_intra_results)
print(hand_intra_results)

hip_inter_results = []

for m in ['1', '2', '3']:
    for method in columns:   # columns = list of method names
        df_temp = df_hip[(df_hip["Measurements"] == m) &
                         (df_hip["Methods"] == method)]
        icc = pg.intraclass_corr(
            data=df_temp,
            targets='Samples',
            raters='Observer',
            ratings='Ratings'
        )
        hip_inter_results.append(icc.assign(Measurement=m, Method=method))

hip_inter_results = pd.concat(hip_inter_results)
# print(hip_inter_results)

for method in columns:
    df = hip_inter_results[hip_inter_results["Method"] == method]
    # print(df)

hip_intra_results = []

for obs in ['A', 'B', 'C', 'D']:
    for method in columns:
        df_temp = df_hip[(df_hip["Observer"] == obs) &
                         (df_hip["Methods"] == method)]
        icc = pg.intraclass_corr(
            data=df_temp,
            targets='Samples',
            raters='Measurements',
            ratings='Ratings'
        )
        hip_intra_results.append(icc.assign(Observer=obs, Method=method))

hip_intra_results = pd.concat(hip_intra_results)
# print(hip_intra_results)

for method in columns:
    df = hip_intra_results[hip_intra_results["Method"] == method]
    # print(df)