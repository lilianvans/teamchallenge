import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
import os
import csv
import pydicom
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares
from pathlib import Path

#make sure to not show a general tkinter window, we only want error/instruction messages
root_tk = tk.Tk()
root_tk.withdraw()

#path to the folders or files that we need
ROOT_DATA_FOLDER = Path(r"ENTER PATH TO HIP FILES")
RESULTS_CSV_FILE = "acetabulum_ellipse_inter_intraobserver_NAME.csv" #change to your name
INSTRUCTION_IMAGES = ["Example_image_hip.jpg", "Example_image_acetabulum.jpg"] 

# HARDCODE LESS: define your range of patients here, H should be included
START_PID = "PATIENT BEGIN" 
END_PID = "PATIENT END"

#put everything together in a class, so that it is clear to use and variables can be used across functions
class AcetabulumEllipseTool:
    def __init__(self, dicom_path, raw_image_data, patient_id, scan_date, serial_num, img_num):
        #make variables that can be used
        self.dicom_path = dicom_path
        self.patient_id = patient_id
        self.scan_date = scan_date
        self.serial_num = serial_num
        self.img_num = img_num
        self.selected_points = []
        
        #showing the image and normalization
        p2, p98 = np.percentile(raw_image_data, (2, 98))
        self.display_image = (raw_image_data - p2) / (p98 - p2)
        self.display_image = np.clip(self.display_image, 0, 1)

        #sizes of the user interface
        self.figure = plt.figure(figsize=(16, 9))
        #divide the interface into a grid
        grid = self.figure.add_gridspec(2, 5, width_ratios=[1.8, 1.2, 1.5, 1.5, 1.5])
        
        #make sure the example images can be visualised
        self.ax_instruction_top = self.figure.add_subplot(grid[0, 0])
        self.ax_instruction_bottom = self.figure.add_subplot(grid[1, 0])
        self.load_reference_guides()

        #instructions to be shown between example and main image
        self.ax_text = self.figure.add_subplot(grid[:, 1])
        self.ax_text.set_axis_off()
        instructions_text = (
            "USER INSTRUCTIONS:\n\n"
            "1. Zoom in to the left acetabulum\n"
            "    using the tool in the top bar.\n\n"
            "    (See 1st example image)\n\n"
            "2. Deselect the zoom tool.\n\n"
            "3. Mark 6 points total along the\n"
            "    edge of the acetabulum.\n"
            "    Points 1 and 6 define the midline.\n\n"
            "4. Click points from Left to Right.\n\n"
            "    (See 2nd example image)\n\n"
            "5. Press 'Enter' to finish and\n"
            "    calculate the ellipse ratio.\n\n"
            "6. Close this window to move\n"
            "    to the next patient.\n\n"
            "--- Press 'r' to reset ---"
        )
        self.ax_text.text(0, 0.95, instructions_text, transform=self.ax_text.transAxes, 
                          fontsize=12, verticalalignment='top', fontweight='medium', linespacing=1.8)

        #make sure the main dicom image is visualized
        self.main_ax = self.figure.add_subplot(grid[:, 2:])
        self.main_ax.imshow(self.display_image, cmap='gray')
        
        #header showing patient id, date, serial and image number
        header_info = f"PID: {patient_id} | DATE: {scan_date} | SERIAL: {serial_num} | IMG: {img_num}"
        self.main_ax.set_title(header_info, fontsize=12, fontweight='bold', pad=25)
        
        #visualizing the points, the curve line and the ellipse
        self.point_plot, = self.main_ax.plot([], [], 'ro', markersize=6, label='Points', zorder=5)
        self.curve_plot, = self.main_ax.plot([], [], 'cyan', linewidth=1, alpha=0.5, zorder=4)
        
        #placeholders for drawings
        self.ellipse_patches = [] 
        self.axis_lines = []
        
        #overlay that shows the eventual results
        self.results_overlay = self.main_ax.text(0.02, 0.95, "", transform=self.main_ax.transAxes, 
                                                color='yellow', fontsize=10, fontweight='bold', verticalalignment='top')

        #make sure that a click is acknowledged and places a point
        self.click_event = self.figure.canvas.mpl_connect('button_press_event', self.handle_click)
        #make sure that the enter key is acknowledged and begins the calculations
        self.key_event = self.figure.canvas.mpl_connect('key_press_event', self.handle_keypress)
        
        #show the interface
        plt.tight_layout()
        plt.show()

    def load_reference_guides(self):
        """
        Function that actually shows the example images on the left side of the interface
        """
        guide_axes = [self.ax_instruction_top, self.ax_instruction_bottom]
        for ax, img_filename in zip(guide_axes, INSTRUCTION_IMAGES):
            ax.set_axis_off()
            if os.path.exists(img_filename):
                img_data = mpimg.imread(img_filename)
                ax.imshow(img_data)
            else:
                ax.text(0.5, 0.5, f"Missing:\n{img_filename}", ha='center')

    def fit_ellipse(self, x_pts, y_pts):
        """
        Fits an ellipse while making a major axis align with the first and last points that were clicked.
        """
        p_start = np.array([x_pts[0], y_pts[0]])
        p_end = np.array([x_pts[-1], y_pts[-1]])
        midline_vec = p_end - p_start
        dist_start_end = np.linalg.norm(midline_vec)
        
        cx0, cy0 = (p_start + p_end) / 2
        a0 = dist_start_end / 2 
        b0 = a0 * 0.5 
        theta0 = np.arctan2(midline_vec[1], midline_vec[0])

        def residuals(params, x, y):
            cx, cy, a, b, theta = params
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            xr = (x - cx) * cos_t + (y - cy) * sin_t
            yr = -(x - cx) * sin_t + (y - cy) * cos_t
            base_resid = (xr/a)**2 + (yr/b)**2 - 1
            
            penalty_a = (a - (dist_start_end / 2)) * 1
            penalty_cx = (cx - cx0) * 0.5
            penalty_cy = (cy - cy0) * 0.5            
            return np.append(base_resid, [penalty_a, penalty_cx, penalty_cy])

        res = least_squares(residuals, [cx0, cy0, a0, b0, theta0], args=(x_pts, y_pts), max_nfev=1000)
        return res.x 

    def draw_ellipse_with_axes(self, params, color, label_prefix="", lw=2):
        """
        Draws the ellipse patch and the a/b axis lines onto the patient image
        """
        cx, cy, a, b, theta = params
        ellipse = Ellipse(xy=(cx, cy), width=a*2, height=b*2, angle=np.degrees(theta), 
                           edgecolor=color, fc='none', lw=lw, ls='--', zorder=6)
        self.main_ax.add_patch(ellipse)
        self.ellipse_patches.append(ellipse)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        ax_line, = self.main_ax.plot([cx - a*cos_t, cx + a*cos_t], [cy - a*sin_t, cy + a*sin_t], 
                                    color=color, lw=1, alpha=0.7, zorder=6)
        bx_line, = self.main_ax.plot([cx + b*sin_t, cx - b*sin_t], [cy - b*cos_t, cy + b*cos_t], 
                                    color=color, lw=1, alpha=0.7, zorder=6)
        self.axis_lines.extend([ax_line, bx_line])

    def handle_click(self, event):
        """
        Saves the coordinates of where a point is put up until 6 points.
        """
        if self.figure.canvas.manager.toolbar.mode != '': return
        if event.inaxes != self.main_ax: return
            
        if len(self.selected_points) < 6:
            self.selected_points.append([event.xdata, event.ydata])
            self.refresh_visuals()

    def handle_keypress(self, event):
        """
        Begins calculations if enter is pressed, and resets everything if r is pressed
        """
        if event.key == 'enter':
            self.process_measurements()
        elif event.key == 'r':
            self.selected_points = []
            for p in self.ellipse_patches: p.remove()
            for l in self.axis_lines: l.remove()
            self.ellipse_patches = []
            self.axis_lines = []
            self.results_overlay.set_text("")
            self.refresh_visuals()

    def refresh_visuals(self):
        """
        Visualising the point on top of the patient image
        """
        if self.selected_points:
            pts_array = np.array(self.selected_points)
            self.point_plot.set_data(pts_array[:, 0], pts_array[:, 1])
        else:
            self.point_plot.set_data([], [])
            self.curve_plot.set_data([], [])
        self.figure.canvas.draw()

    def process_measurements(self):
        """
        Interpolates points to a line and fits an ellipse.
        """
        try:
            if len(self.selected_points) != 6:
                messagebox.showerror("Point Error", "Please select exactly 6 points.")
                return
                
            pts = np.array(self.selected_points)
            t = np.linspace(0, 1, len(pts))
            t_smooth = np.linspace(0, 1, 100)
            poly_x = np.poly1d(np.polyfit(t, pts[:, 0], 2))
            poly_y = np.poly1d(np.polyfit(t, pts[:, 1], 2))
            x_line, y_line = poly_x(t_smooth), poly_y(t_smooth)
            self.curve_plot.set_data(x_line, y_line)
            
            res_full = self.fit_ellipse(x_line, y_line)
            ratio_full = min(abs(res_full[2]), abs(res_full[3])) / max(abs(res_full[2]), abs(res_full[3]))
            
            res_l = self.fit_ellipse(pts[0:3, 0], pts[0:3, 1])
            ratio_l = min(abs(res_l[2]), abs(res_l[3])) / max(abs(res_l[2]), abs(res_l[3]))

            res_r = self.fit_ellipse(pts[3:6, 0], pts[3:6, 1])
            ratio_r = min(abs(res_r[2]), abs(res_r[3])) / max(abs(res_r[2]), abs(res_r[3]))

            vec1 = pts[2] - pts[0]
            vec2 = pts[5] - pts[3]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            for p in self.ellipse_patches: p.remove()
            for l in self.axis_lines: l.remove()
            self.ellipse_patches, self.axis_lines = [], []

            self.draw_ellipse_with_axes(res_full, 'cyan', lw=2.5)
            self.draw_ellipse_with_axes(res_l, 'yellow', lw=1)
            self.draw_ellipse_with_axes(res_r, 'magenta', lw=1)
            
            result_string = (f"Full Ratio: {ratio_full:.3f}\n"
                             f"Angle: {angle_deg:.1f}°\n"
                             f"Ratio Left (1-3): {ratio_l:.3f}\n"
                             f"Ratio Right (4-6): {ratio_r:.3f}")
            self.results_overlay.set_text(result_string)
            
            #save all the data into a CSV file with Serial and Image Number
            file_exists = os.path.isfile(RESULTS_CSV_FILE)
            with open(RESULTS_CSV_FILE, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["Patient_ID", "Date", "Serial_Number", "Image_Number", "Ratio_Full", "Angle", "Ratio_L", "Ratio_R"])
                writer.writerow([self.patient_id, self.scan_date, self.serial_num, self.img_num, 
                                 round(ratio_full, 4), round(angle_deg, 2), round(ratio_l, 4), round(ratio_r, 4)])
            
            self.figure.canvas.draw()
            messagebox.showinfo("Success", f"Data saved for {self.patient_id}")
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred: {str(e)}")

#run the main loop to execute everything and repeat for all images and patients
if __name__ == "__main__":
    if ROOT_DATA_FOLDER.is_dir():
        #build index of all DICOM images
        index = {}
        for file_path in ROOT_DATA_FOLDER.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                pid = str(ds.get("PatientID", "Unknown"))
                study_date = ds.get("StudyDate", "Unknown")
                
                # Logic to get the unique serial folder name and subfolder
                parent_1 = os.path.dirname(file_path) # DICOM folder
                img_subfolder = os.path.basename(os.path.dirname(parent_1)) # 1, 2, or 3
                serial_folder = os.path.basename(os.path.dirname(os.path.dirname(parent_1))) # H42_20241101_21776 folder
                
                # Split out the serial number (e.g. 21776) from the end of the folder name
                serial_num = serial_folder.split("_")[-1] if "_" in serial_folder else serial_folder
                
                if pid not in index: index[pid] = {}
                if study_date not in index[pid]: index[pid][study_date] = {}
                if serial_num not in index[pid][study_date]: index[pid][study_date][serial_num] = {}
                
                # Store the first valid file per unique subfolder
                if img_subfolder not in index[pid][study_date][serial_num]:
                    index[pid][study_date][serial_num][img_subfolder] = file_path
            except: continue

        #filter patients by range
        sorted_pids = sorted(index.keys())
        try:
            target_pids = sorted_pids[sorted_pids.index(START_PID) : sorted_pids.index(END_PID) + 1]
        except ValueError:
            print(f"Start or end patient ID not found")
            target_pids = []

        #iterate through our structured index
        for pid in target_pids:
            for date_raw in sorted(index[pid].keys()):
                for sn in sorted(index[pid][date_raw].keys()):
                    for img_n in sorted(index[pid][date_raw][sn].keys()):
                        full_path = str(index[pid][date_raw][sn][img_n])
                        
                        #format date for display
                        try:
                            scan_dt = datetime.strptime(date_raw, "%Y%m%d").strftime("%d-%m-%Y")
                        except:
                            scan_dt = date_raw

                        try:
                            #read the image
                            dicom_data = sitk.ReadImage(full_path)
                            pixel_array = sitk.GetArrayFromImage(dicom_data).astype(float)
                            if pixel_array.ndim == 3: pixel_array = pixel_array[0]
                            
                            #run our main hip measurement tool
                            print(f"Starting Patient: {pid} | Serial: {sn} | Image: {img_n}")
                            AcetabulumEllipseTool(full_path, pixel_array, pid, scan_dt, sn, img_n)
                        except Exception as error:
                            messagebox.showerror("File Error", f"Error: {str(error)}")
    else:
        messagebox.showerror("Path Error", f"Not found: {ROOT_DATA_FOLDER}")

print("Processing finished for the selected range.")
