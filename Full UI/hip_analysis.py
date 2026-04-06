from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg", force=True)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares

from dicom_utils import WorkflowError


INSTRUCTION_IMAGES = [
    "Example_image_hip.jpg",
    "Example_image_acetabulum.jpg",
]


class AcetabulumEllipseTool:
    def __init__(
        self,
        dicom_path,
        raw_image_data,
        patient_id,
        scan_date,
        image_label,
        output_dir,
    ):
        self.dicom_path = Path(dicom_path)
        self.patient_id = patient_id
        self.scan_date = scan_date
        self.image_label = image_label
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selected_points = []
        self.finished_successfully = False
        self.results = None
        self.point_labels = []
        self.point_artists = []
        self.ellipse_patches = []
        self.axis_lines = []

        # show image with percentile normalization so contrast is easier to see
        p2, p98 = np.percentile(raw_image_data, (2, 98))
        denom = max(p98 - p2, 1e-8)
        self.display_image = (raw_image_data - p2) / denom
        self.display_image = np.clip(self.display_image, 0, 1)

        # build the interface layout
        self.figure = plt.figure(figsize=(16, 9))
        grid = self.figure.add_gridspec(2, 5, width_ratios=[1.8, 1.2, 1.5, 1.5, 1.5])

        # example images on the left
        self.ax_instruction_top = self.figure.add_subplot(grid[0, 0])
        self.ax_instruction_bottom = self.figure.add_subplot(grid[1, 0])
        self.load_reference_guides()

        # written instructions in the middle
        self.ax_text = self.figure.add_subplot(grid[:, 1])
        self.ax_text.set_axis_off()
        instructions_text = (
            "USER INSTRUCTIONS:\n\n"
            "1. Zoom in to the left acetabulum\n"
            "    using the tool in the top bar.\n\n"
            "    (See 1st example image)\n\n"
            "2. Deselect the zoom tool.\n\n"
            "3. Press 'Enter' once to start\n"
            "    point placement.\n\n"
            "4. Mark 6 points total along the\n"
            "    edge of the acetabulum.\n"
            "    Points 1 and 6 define the midline.\n\n"
            "5. Click points from Left to Right.\n\n"
            "    (See 2nd example image)\n\n"
            "6. After all 6 points are placed,\n"
            "    press 'Enter' again to calculate.\n\n"
            "7. Press 'r' to reset.\n\n"
            "8. Close this window to return\n"
            "    to the main UI.\n"
        )
        self.ax_text.text(
            0,
            0.95,
            instructions_text,
            transform=self.ax_text.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontweight="medium",
            linespacing=1.8,
        )

        # main patient image on the right
        self.main_ax = self.figure.add_subplot(grid[:, 2:])
        self.main_ax.imshow(self.display_image, cmap="gray")

        # header with patient information
        header_info = f"PID: {patient_id} | DATE: {scan_date}"
        self.main_ax.set_title(header_info, fontsize=12, fontweight="bold", pad=25)

        # overlay used for instructions and final results
        self.results_overlay = self.main_ax.text(
            0.02,
            0.95,
            "Zoom if needed, then turn zoom off and press Enter to start.",
            transform=self.main_ax.transAxes,
            color="yellow",
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
        )

        # only reset is handled directly by the keypress callback
        self.figure.canvas.mpl_connect("key_press_event", self.handle_keypress)

        plt.tight_layout()
        plt.show(block=False)

        pts = self.collect_points()
        if pts is None:
            plt.close(self.figure)
            return

        self.selected_points = pts.tolist()

        # run the calculation after the second Enter
        self.process_measurements(pts)
        self.finished_successfully = True

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        plt.show(block=True)

    def load_reference_guides(self):
                # Show the two example images on the left side of the interface.

        guide_axes = [self.ax_instruction_top, self.ax_instruction_bottom]
        script_dir = Path(__file__).resolve().parent
        headings = ["Example 1: Where to Zoom", "Example 2: Point Placement"]

        for ax, img_filename, heading in zip(guide_axes, INSTRUCTION_IMAGES, headings):
            ax.set_axis_off()
            img_path = script_dir / img_filename
            if img_path.exists():
                img_data = mpimg.imread(img_path)
                ax.imshow(img_data)
            else:
                ax.text(0.5, 0.5, f"Missing:\n{img_filename}", ha="center", va="center")
            ax.text(
                0.5,
                1.02,
                heading,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    def collect_points(self):
                # Wait for:
        # 1. first Enter to begin point placement
        # 2. six clicks from left to right
        # 3. second Enter to run the calculation

        self.results_overlay.set_text(
            "Zoom to the acetabulum first.\n"
            "Then turn zoom OFF and press Enter to start point placement."
        )
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        # wait for the first Enter
        start_ready = {"go": False}

        def wait_for_first_enter(event):
            if event.key in ("enter", "return"):
                start_ready["go"] = True

        cid_start = self.figure.canvas.mpl_connect("key_press_event", wait_for_first_enter)

        while not start_ready["go"] and plt.fignum_exists(self.figure.number):
            plt.pause(0.05)

        self.figure.canvas.mpl_disconnect(cid_start)

        if not plt.fignum_exists(self.figure.number):
            return None

        self.results_overlay.set_text("Now click 6 points from left to right.")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        accepted_points = []

        while len(accepted_points) < 6 and plt.fignum_exists(self.figure.number):
            pts = plt.ginput(n=1, timeout=-1, show_clicks=False)

            if len(pts) == 0:
                self.results_overlay.set_text("Point selection cancelled.")
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                return None

            x, y = pts[0]
            accepted_points.append([x, y])

            point_artist, = self.main_ax.plot(
                x,
                y,
                marker="o",
                markersize=10,
                markerfacecolor="red",
                markeredgecolor="yellow",
                markeredgewidth=1.5,
                linestyle="None",
                zorder=10,
            )
            self.point_artists.append(point_artist)

            txt = self.main_ax.text(
                x + 6,
                y - 6,
                str(len(accepted_points)),
                color="yellow",
                fontsize=11,
                fontweight="bold",
                zorder=11,
            )
            self.point_labels.append(txt)

            if len(accepted_points) < 6:
                self.results_overlay.set_text(f"Placed point {len(accepted_points)} of 6")
            else:
                self.results_overlay.set_text("6 points placed. Press Enter to calculate.")

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        if len(accepted_points) != 6:
            return None

        # wait for the second Enter
        calc_ready = {"go": False}

        def wait_for_second_enter(event):
            if event.key in ("enter", "return"):
                calc_ready["go"] = True

        cid_calc = self.figure.canvas.mpl_connect("key_press_event", wait_for_second_enter)

        while not calc_ready["go"] and plt.fignum_exists(self.figure.number):
            plt.pause(0.05)

        self.figure.canvas.mpl_disconnect(cid_calc)

        if not plt.fignum_exists(self.figure.number):
            return None

        self.results_overlay.set_text("Calculating...")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        return np.array(accepted_points, dtype=float)

    def fit_ellipse(self, x_pts, y_pts):
                # Fit an ellipse while making the major axis stay aligned
        # with the first and last selected points.

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
            a = max(abs(a), 1e-6)
            b = max(abs(b), 1e-6)

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            xr = (x - cx) * cos_t + (y - cy) * sin_t
            yr = -(x - cx) * sin_t + (y - cy) * cos_t
            base_resid = (xr / a) ** 2 + (yr / b) ** 2 - 1

            # keep the fit close to the clicked midline and centre
            penalty_a = (a - (dist_start_end / 2)) * 1.0
            penalty_cx = (cx - cx0) * 0.5
            penalty_cy = (cy - cy0) * 0.5
            return np.append(base_resid, [penalty_a, penalty_cx, penalty_cy])

        res = least_squares(
            residuals,
            [cx0, cy0, a0, b0, theta0],
            args=(x_pts, y_pts),
            max_nfev=1000,
        )
        return res.x

    def draw_ellipse_with_axes(self, params, color, lw=2):
                # Draw the ellipse and its a/b axes on top of the patient image.

        cx, cy, a, b, theta = params
        a = abs(a)
        b = abs(b)

        ellipse = Ellipse(
            xy=(cx, cy),
            width=a * 2,
            height=b * 2,
            angle=np.degrees(theta),
            edgecolor=color,
            fc="none",
            lw=lw,
            ls="--",
            zorder=6,
        )
        self.main_ax.add_patch(ellipse)
        self.ellipse_patches.append(ellipse)

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        ax_line, = self.main_ax.plot(
            [cx - a * cos_t, cx + a * cos_t],
            [cy - a * sin_t, cy + a * sin_t],
            color=color,
            lw=1,
            alpha=0.7,
            zorder=6,
        )
        bx_line, = self.main_ax.plot(
            [cx + b * sin_t, cx - b * sin_t],
            [cy - b * cos_t, cy + b * cos_t],
            color=color,
            lw=1,
            alpha=0.7,
            zorder=6,
        )
        self.axis_lines.extend([ax_line, bx_line])

    def handle_keypress(self, event):
                # Reset the current drawing if r is pressed.

        if event.key == "r":
            self.selected_points = []
            self.finished_successfully = False
            self.results = None

            for p in self.ellipse_patches:
                try:
                    p.remove()
                except Exception:
                    pass
            for l in self.axis_lines:
                try:
                    l.remove()
                except Exception:
                    pass
            for txt in self.point_labels:
                try:
                    txt.remove()
                except Exception:
                    pass
            for artist in self.point_artists:
                try:
                    artist.remove()
                except Exception:
                    pass

            self.ellipse_patches = []
            self.axis_lines = []
            self.point_labels = []
            self.point_artists = []

            self.main_ax.clear()
            self.main_ax.imshow(self.display_image, cmap="gray")

            header_info = f"PID: {self.patient_id} | DATE: {self.scan_date}"
            self.main_ax.set_title(header_info, fontsize=12, fontweight="bold", pad=25)

            self.results_overlay = self.main_ax.text(
                0.02,
                0.95,
                "Reset complete. Close and reopen analysis to start again.",
                transform=self.main_ax.transAxes,
                color="yellow",
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
            )

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def _save_main_axis_only(self, output_path: Path):
        # save only the main analysis image, not the full instruction layout
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()

        bbox = self.main_ax.get_tightbbox(renderer).expanded(1.02, 1.02)
        bbox_inches = bbox.transformed(self.figure.dpi_scale_trans.inverted())

        self.figure.savefig(
            output_path,
            dpi=150,
            bbox_inches=bbox_inches,
        )

    def process_measurements(self, pts):
                # Interpolate the six clicked points to a smooth curve,
        # then fit full, left, and right ellipses.

        t = np.linspace(0, 1, len(pts))
        t_smooth = np.linspace(0, 1, 100)

        # smooth the clicked points with low-order polynomials
        poly_x = np.poly1d(np.polyfit(t, pts[:, 0], 2))
        poly_y = np.poly1d(np.polyfit(t, pts[:, 1], 2))
        x_line = poly_x(t_smooth)
        y_line = poly_y(t_smooth)

        self.main_ax.plot(
            x_line,
            y_line,
            color="cyan",
            linewidth=1,
            alpha=0.5,
            zorder=4,
        )

        # fit full ellipse to the smoothed curve
        res_full = self.fit_ellipse(x_line, y_line)
        ratio_full = min(abs(res_full[2]), abs(res_full[3])) / max(abs(res_full[2]), abs(res_full[3]))

        # fit smaller ellipses to the left and right point groups
        res_l = self.fit_ellipse(pts[0:3, 0], pts[0:3, 1])
        ratio_l = min(abs(res_l[2]), abs(res_l[3])) / max(abs(res_l[2]), abs(res_l[3]))

        res_r = self.fit_ellipse(pts[3:6, 0], pts[3:6, 1])
        ratio_r = min(abs(res_r[2]), abs(res_r[3])) / max(abs(res_r[2]), abs(res_r[3]))

        # angle between the two outer edge directions
        vec1 = pts[2] - pts[0]
        vec2 = pts[5] - pts[3]

        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            angle_deg = float("nan")
        else:
            cos_theta = np.dot(vec1, vec2) / denom
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        for p in self.ellipse_patches:
            try:
                p.remove()
            except Exception:
                pass
        for l in self.axis_lines:
            try:
                l.remove()
            except Exception:
                pass
        self.ellipse_patches = []
        self.axis_lines = []

        self.draw_ellipse_with_axes(res_full, "cyan", lw=2.5)
        self.draw_ellipse_with_axes(res_l, "yellow", lw=1)
        self.draw_ellipse_with_axes(res_r, "magenta", lw=1)

        result_string = (
            f"Full Ratio: {ratio_full:.3f}\n"
            f"Angle: {angle_deg:.1f}°\n"
            f"Ratio Left (1-3): {ratio_l:.3f}\n"
            f"Ratio Right (4-6): {ratio_r:.3f}"
        )
        self.results_overlay.set_text(result_string)

        preview_path = self.output_dir / "hip_analysis_preview.png"
        self._save_main_axis_only(preview_path)

        self.results = {
            "ratio_full": round(float(ratio_full), 4),
            "angle_deg": round(float(angle_deg), 2),
            "ratio_left": round(float(ratio_l), 4),
            "ratio_right": round(float(ratio_r), 4),
            "selected_points": pts.tolist(),
            "preview_path": str(preview_path),
        }

        self.figure.canvas.draw_idle()


def run_hip_acetabulum_analysis(
    dicom_path: Path,
    output_dir: Path,
    patient_id: str = "",
    scan_date: str = "",
    image_label: str = "",
) -> dict | None:
    try:
        dicom_data = sitk.ReadImage(str(dicom_path))
        pixel_array = sitk.GetArrayFromImage(dicom_data).astype(float)

        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        tool = AcetabulumEllipseTool(
            dicom_path=dicom_path,
            raw_image_data=pixel_array,
            patient_id=patient_id,
            scan_date=scan_date,
            image_label=image_label,
            output_dir=output_dir,
        )

        if tool.results is None:
            return None

        return tool.results

    except Exception as e:
        raise WorkflowError(f"Hip analysis could not be completed: {e}")
