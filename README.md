# TEAM CHALLENGE README
This repository contains all the scripts for the UI, and the basic scripts that were constructed before these were integrated in the UI structure. This repository was made for the course Team Challenge at the TU/e and UU.  

**Full UI:**  
Contains the complete UI that can be used for the analysis. The UI has a separate README, which is in this folder.

**Images:**  
The example images used for the Acetabulum_curvature_semi_automatic.py interface.

**finger_segmentation_separate.py:**  
Can semi-automatically segment a separate finger on a DICOM x-ray image using a lasso approach. Saves segmentation masks of the different finger bones.

**finger_segmentation_batch.py:**  
Can semi-automatically segment a finger on multiple scans of different patients using a lasso approach. Saves segmentation masks of the different finger bones.

**Acetabulum_curvature_semi_automatic.py:**  
The script that uses a semi-automatic approach for extracting 4 different hip parameters. Points need to be marked by the user. 

**Finger_curvature_automatic.py:**  
The script that can automatically analyze joint angles and total flexion of a finger using segmentation masks as input.

**ratiometacarpal.py:**  
Can automatically compute several length-width and width-width ratios of the finger bones using segmentation masks as input.

**result_analysis:**    
Extracts patient characteristics based on the DICOM metadata, folder names & the original excel data sheet.

**feature_selection_hand.py:**  
This script compares the data sheets of the hand parameters of patients and healthy controls to find the significantly progressing parameters (trends) in an age-matched manner. 

**general_disease_metric.py:**    
Builds a general disease metric from the 6 features with the lowest p-value from feature_selection_hand.py. This metric could be used to quantify general disease progression in the hands of patients. This as well is done in a age-matched manner, comparing to a healthy baseline.

**hip_grouped_analysis.py:**  
Maps out the different hip parameters in patients over age in a scatter plot. Finds a trend line and assesses significance of the trend for the full group.
