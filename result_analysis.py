from pathlib import Path
import pydicom

import pandas as pd  
# root map
root = Path(r"bekken AP")

# index and rows for Excel
index = {}
rows = []

def convert_to_months(age_str):
    if age_str is None:
        return None

    try:
        time_span = age_str[-1]
        age_value = age_str[:-1]

        if not age_value.isdigit():
            return None

        age_int = int(age_value)

        if time_span == "Y":
            return float(age_int * 12)
        elif time_span == "M":
            return float(age_int)
        elif time_span == "W":
            return float(age_int / 52 * 12)
        elif time_span == "D":
            return float(age_int / 365 * 12)
        else:
            return None
    except Exception:
        return None




# loop over files in root map
for file in root.rglob("*"):
    if not file.is_file():
        continue

    try:
        ds = pydicom.dcmread(file, stop_before_pixels=True)
    except Exception:
        continue

    folder_name = file.parent.parent.name

    # get metadata
    patient_id = ds.get("PatientID")
    study_date = ds.get("StudyDate")
    patient_age = ds.get("PatientAge")
    body_part = ds.get("BodyPartExamined")
    laterality= ds.get("ImageLaterality")
    

    # check data exists
    if patient_id and study_date and patient_age and body_part and folder_name:

        # convert age to months
        patient_age_months = convert_to_months(patient_age)

        # add to list
        rows.append({
            "Patient_ID": patient_id,
            "StudyDate": study_date,
            "Folder": folder_name,
            "Patient_Age_Months": patient_age_months,
            "BodyPart": body_part,
            "Laterality": laterality,
            
        })

    


# print overview
for patient_id, studies in index.items():
    print(f'Patient {patient_id}:')
    for study_date, series in studies.items():
        print(f'    {study_date}: {[desc for desc in series.keys()]}')


# make dataframe
df = pd.DataFrame(rows)

# sort dataframe
df = df.sort_values(by=["Patient_ID", "StudyDate"])

# save to excel
output_path = "hip_overzicht.xlsx"
df.to_excel(output_path, index=False)
print(f"Excel saved as: {output_path}")

