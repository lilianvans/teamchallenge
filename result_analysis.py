import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import filters, exposure, morphology, feature
from scipy import ndimage as ndi
from pathlib import Path
import pydicom
import cv2

def convert_to_months(age_str):
    time_span = age_str[-1]
    age_str = age_str[:-1]
    if time_span == "Y":
        age_int = int(age_str)
        age_fl = float(age_int*12)
    elif time_span == "M":
        age_fl = float(age_str)
    elif time_span == "W":
        age_int = int(age_str)
        age_fl = float(age_int/52*12)
    elif time_span == "D":
        age_int = int(age_str)
        age_fl = float(age_int/365*12)
    else:
        print("Could not find time span")
    return age_fl

root = Path(r"Hand Hurler")
# Build an index of all DICOM images, sorted by patient, study date and image series description.
index = {}
patient_age_distribution = []
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
    patient_age = ds.get("PatientAge")
    series_description = ds.get("SeriesDescription")
    
    # Create new index levels.
    if patient_age is not None:
        if patient_id not in index:
            index[patient_id] = {}
        if study_date not in index[patient_id]:
            index[patient_id][study_date] = {}
        if series_description not in index[patient_id][study_date]: 
            index[patient_id][study_date][series_description] = []

        # Convert age to a float of months
        patient_age = convert_to_months(patient_age)
        patient_age_distribution.append(patient_age)
        # Add the file to the index.
        index[patient_id][study_date][series_description].append(patient_age)
        index[patient_id][study_date][series_description].append(file)


    # Display an overview of all indexed DICOM images.
for patient_id, studies in index.items():
    print(f'Patient {patient_id}:')
    for study_date, series in studies.items():
        print(f'    {study_date}: {[description for description in series.keys()]}')
# Read one of the images.
reader = sitk.ImageSeriesReader()
print(index['H44']['20150901']['Hand PA'][0])
age, file_read = index['H44']['20150901']['Hand PA']
# print(Path(file_read))
img = sitk.ReadImage(file_read)
# img = reader.Execute()

print('age', age)
# Show the image.
img_array = sitk.GetArrayFromImage(img)

print(min(patient_age_distribution), max(patient_age_distribution))


# Convert to numpy array
arr = np.array(patient_age_distribution)

# Create histogram
plt.hist(arr, bins=200, range=(0.5, 300), edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()


