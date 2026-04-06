Main.py 
-	Starts the PyQt application and opens launch window, where user chooses between the hand and hip interfaces

Ui_launcher.py
-	Creates the small starting screen where buttons for hand and hip interfaces open the corresponding interfaces

Hand_ui.py 
-	Includes the specific hand workflow. Loads to UI and locks into hand analysis pipeline to avoid overlap 

Hip_ui.py
-	Includes the specific hip workflow. Loads to UI and locks into hip analysis pipeline to avoid overlap 

Workflow_ui.py
-	Contains the main UI and overall workflow
-	Patient selection, DICOM import, preview tabs, buttons and loading of results for both hand and hip interfaces are handled here 

Dicom_utils.py 
-	Contains helper functions for anything related to DICOMs
-	Generates patient folders, manages patient age and creates the preview images

Segmentation.py 
-	Includes the lasso-based bone segmentation
-	User is asked to manually outline the bone and script then selects the masks to be saved

Measurements.py
-	Analysis for hand segmentation masks
-	Calculates finger ratio and curvature calculations

Hand_reporting.py
-	Builds the results and converts to CSV for exporting 

Hip_analysis.py
-	Uses the semi-automatic acetabulum analysis 
-	User places size points and then ellipse fitting is used to calculate measurements 

Hip_reporting.py
-	Builds the results and converts to CSV for exporting 

Progression_plots.py
-	Creates the hand progression plots from the saved measurements
-	Includes comparison with reference data, outlier removal and generates visual plot for progression trend Hurler compared to healthy patient baseline

Hip_progression_plots.py 
-	Creates the hip progression plots from the saved measurements
-	Loads patient data and creates visual plot for hip measurement changes over time

Reporting.py
-	Includes the file saving helper functions for both exports
-	Including CSV writing, timestamp handling, etc 
