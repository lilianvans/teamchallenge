import json
from pathlib import Path

import pydicom
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dicom_utils import (
    WorkflowError,
    copy_dicom_to_study,
    ensure_patient_structure,
    ensure_study_structure,
    get_bone_age_months_from_dicom,
    get_patient_age_months_from_dicom,
    get_patient_age_years_from_dicom,
    get_patient_display_from_dicom,
    get_patient_folder_name_from_dicom,
    get_study_date_from_dicom,
    get_study_display_from_dicom,
    get_study_folder_name_from_dicom,
    get_study_year_from_dicom,
    load_dicom_for_preview,
)
from hand_reporting import (
    build_hand_results,
    build_hand_text_report,
    save_hand_results_csv,
    update_hand_patient_analysis_files,
)
from hip_analysis import run_hip_acetabulum_analysis
from hip_reporting import (
    build_hip_results,
    build_hip_text_report,
    save_hip_results_csv,
)
from measurements import calculate_curvature, calculate_finger_ratios
from reporting import make_results_json_safe, save_json, save_report
from segmentation import run_lasso_segmentation


HAND_INSTRUCTIONS = (
    "1. Open DICOM or folder.\n"
    "2. Select image.\n"
    "3. Choose patient and import.\n"
    "4. Reopen saved studies by clicking a patient.\n"
    "5. Filter by year if needed.\n"
    "6. Run segmentation -> ratios -> curvature.\n"
    "7. q finishes segmentation window.\n"
    "8. Analysis tab = segmentation/measurement previews.\n"
    "9. Results tab = hand JSON + progression plots.\n"
)

HIP_INSTRUCTIONS = (
    "1. Open DICOM or folder.\n"
    "2. Select image.\n"
    "3. Choose patient and import.\n"
    "4. Run hip acetabulum analysis.\n"
    "5. Place 6 points from left to right.\n"
    "6. Press Enter to calculate and save preview.\n"
    "7. Generate hip progression plots after multiple studies exist.\n"
    "8. Save final hip report when finished.\n"
)


# convert a numpy image array into a Qt pixmap so it can be shown in the UI preview labels
def numpy_to_qpixmap(image_8bit):
    if image_8bit.ndim == 2:
        h, w = image_8bit.shape
        bytes_per_line = w
        qimg = QImage(image_8bit.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    if image_8bit.ndim == 3 and image_8bit.shape[2] == 3:
        h, w, _ = image_8bit.shape
        bytes_per_line = 3 * w
        qimg = QImage(image_8bit.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    raise WorkflowError("Unsupported image shape for preview.")


class BaseWorkflowWindow(QMainWindow):
    def __init__(
        self,
        base_dir: Path,
        analysis_type: str = "hand",
        lock_analysis_type: bool = False,
    ) -> None:
        super().__init__()

        self.base_dir = Path(base_dir)
        self.default_analysis_type = analysis_type.strip().lower()
        self.lock_analysis_type = lock_analysis_type

        self.setWindowTitle("Analysis Workflow")
        self.resize(1480, 900)

        self.current_patient_dir: Path | None = None
        self.current_study_dir: Path | None = None
        self.current_import_dicom_path: Path | None = None
        self.current_dicom_path: Path | None = None
        self.current_patient_display: str = ""
        self.current_study_display: str = ""
        self.current_preview_array = None
        self.current_analysis_pixmap: QPixmap | None = None
        self.current_results_pixmap: QPixmap | None = None
        self.current_opened_study_files: list[Path] = []
        self.current_saved_studies_by_year: dict[str, list[Path]] = {}
        self.latest_results: dict | None = None
        self.last_analysis_type: str | None = None
        self.current_list_mode: str = "saved_studies"

        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._apply_initial_analysis_type()
        self.refresh_patient_list()
        self.on_analysis_type_changed()

    # build the full main window layout
    # left side = workflow controls
    # right side = previews and result tabs
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.left_scroll_content = QWidget()
        left_panel = QVBoxLayout(self.left_scroll_content)
        left_panel.setContentsMargins(6, 6, 6, 6)
        left_panel.setSpacing(8)
        self.left_scroll.setWidget(self.left_scroll_content)

        self.right_panel_widget = QWidget()
        right_panel = QVBoxLayout(self.right_panel_widget)
        right_panel.setContentsMargins(6, 6, 6, 6)
        right_panel.setSpacing(8)

        self.main_splitter.addWidget(self.left_scroll)
        self.main_splitter.addWidget(self.right_panel_widget)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setSizes([300, 980])
        self.main_splitter.setHandleWidth(12)

        # patient section
        # this handles importing into an existing patient folder or opening saved patient data
        patient_group = QGroupBox("Patients")
        patient_layout = QVBoxLayout(patient_group)
        patient_form = QFormLayout()

        self.detected_patient_label = QLabel("No DICOM opened yet")
        self.detected_patient_label.setWordWrap(True)
        patient_form.addRow("Detected:", self.detected_patient_label)

        self.patient_choice = QComboBox()
        self.patient_choice.setEditable(False)
        patient_form.addRow("Import into:", self.patient_choice)

        self.selected_patient_label = QLabel("No patient selected")
        self.selected_patient_label.setWordWrap(True)
        patient_form.addRow("Current:", self.selected_patient_label)
        patient_layout.addLayout(patient_form)

        patient_button_row = QHBoxLayout()
        self.assign_patient_btn = QPushButton("Import Image")
        self.assign_patient_btn.clicked.connect(self.assign_patient_and_study)
        patient_button_row.addWidget(self.assign_patient_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_patient_list)
        patient_button_row.addWidget(self.refresh_btn)
        patient_layout.addLayout(patient_button_row)

        self.patient_list = QListWidget()
        self.patient_list.itemClicked.connect(self.load_selected_patient)
        self.patient_list.setMaximumHeight(110)
        patient_layout.addWidget(self.patient_list)

        left_panel.addWidget(patient_group)

        # study and DICOM section
        # this shows the currently opened image and saved studies grouped by year
        dicom_group = QGroupBox("Studies / Images")
        dicom_layout = QVBoxLayout(dicom_group)

        self.image_path_label = QLabel("No DICOM selected")
        self.image_path_label.setWordWrap(True)
        dicom_layout.addWidget(self.image_path_label)

        self.study_assignment_label = QLabel("No saved study selected")
        self.study_assignment_label.setWordWrap(True)
        dicom_layout.addWidget(self.study_assignment_label)

        dicom_button_row = QHBoxLayout()
        self.select_image_btn = QPushButton("Open DICOM")
        self.select_image_btn.clicked.connect(self.select_dicom)
        dicom_button_row.addWidget(self.select_image_btn)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_dicom_folder)
        dicom_button_row.addWidget(self.open_folder_btn)
        dicom_layout.addLayout(dicom_button_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Year:"))
        self.year_choice = QComboBox()
        self.year_choice.currentIndexChanged.connect(self.on_year_changed)
        filter_row.addWidget(self.year_choice)
        dicom_layout.addLayout(filter_row)

        self.study_list = QListWidget()
        self.study_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.study_list.itemClicked.connect(self.on_study_list_item_clicked)
        self.study_list.setMaximumHeight(130)
        dicom_layout.addWidget(self.study_list)

        left_panel.addWidget(dicom_group)

        # analysis branch selector
        # hand and hip share the same base UI but activate different workflow buttons
        analysis_type_group = QGroupBox("Analysis Type")
        analysis_type_layout = QVBoxLayout(analysis_type_group)
        self.analysis_type_choice = QComboBox()
        self.analysis_type_choice.addItems(["Hand", "Hip"])
        self.analysis_type_choice.currentIndexChanged.connect(self.on_analysis_type_changed)
        analysis_type_layout.addWidget(self.analysis_type_choice)
        left_panel.addWidget(analysis_type_group)

        # hand workflow buttons run segmentation, ratio, curvature and progression analysis
        self.hand_workflow_group = QGroupBox("Hand Workflow")
        hand_workflow_layout = QVBoxLayout(self.hand_workflow_group)

        self.segment_btn = QPushButton("1. Segmentation")
        self.segment_btn.clicked.connect(self.run_segmentation_step)
        hand_workflow_layout.addWidget(self.segment_btn)

        self.ratio_btn = QPushButton("2. Finger Ratios")
        self.ratio_btn.clicked.connect(self.run_ratio_step)
        hand_workflow_layout.addWidget(self.ratio_btn)

        self.curvature_btn = QPushButton("3. Curvature")
        self.curvature_btn.clicked.connect(self.run_curvature_step)
        hand_workflow_layout.addWidget(self.curvature_btn)

        self.full_pipeline_btn = QPushButton("Run Full Hand Workflow")
        self.full_pipeline_btn.clicked.connect(self.run_full_workflow)
        hand_workflow_layout.addWidget(self.full_pipeline_btn)

        self.save_report_btn = QPushButton("Save Final Hand Report")
        self.save_report_btn.clicked.connect(self.save_final_report)
        hand_workflow_layout.addWidget(self.save_report_btn)

        self.progression_btn = QPushButton("Generate Hand Progression Plots")
        self.progression_btn.clicked.connect(self.generate_progression_plots_step)
        hand_workflow_layout.addWidget(self.progression_btn)

        left_panel.addWidget(self.hand_workflow_group)

        # hip workflow buttons run the acetabulum analysis and hip progression plots
        self.hip_workflow_group = QGroupBox("Hip Workflow")
        hip_workflow_layout = QVBoxLayout(self.hip_workflow_group)

        self.hip_analysis_btn = QPushButton("1. Run Hip Acetabulum Analysis")
        self.hip_analysis_btn.clicked.connect(self.run_hip_analysis_step)
        hip_workflow_layout.addWidget(self.hip_analysis_btn)

        self.hip_progression_btn = QPushButton("2. Generate Hip Progression Plots")
        self.hip_progression_btn.clicked.connect(self.generate_progression_plots_step)
        hip_workflow_layout.addWidget(self.hip_progression_btn)

        self.hip_save_report_btn = QPushButton("Save Final Hip Report")
        self.hip_save_report_btn.clicked.connect(self.save_final_report)
        hip_workflow_layout.addWidget(self.hip_save_report_btn)

        left_panel.addWidget(self.hip_workflow_group)

        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)

        self.instructions_box = QTextEdit()
        self.instructions_box.setReadOnly(True)
        self.instructions_box.setMinimumHeight(90)
        self.instructions_box.setMaximumHeight(120)
        self.instructions_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        instructions_layout.addWidget(self.instructions_box)

        left_panel.addWidget(instructions_group)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(50)
        self.status_box.setMaximumHeight(70)
        left_panel.addWidget(self.status_box)

        left_panel.addStretch(1)

        # right side tabs
        # DICOM Preview = raw image
        # Analysis Preview = latest segmentation or measurement figure
        # Results = progression plots or final result previews
        self.right_tabs = QTabWidget()
        right_panel.addWidget(self.right_tabs)

        preview_tab = QWidget()
        preview_tab_layout = QVBoxLayout(preview_tab)

        preview_group = QGroupBox("DICOM Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.image_preview = QLabel("No DICOM loaded")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(420)
        self.image_preview.setStyleSheet("border: 1px solid gray; background-color: white;")
        preview_layout.addWidget(self.image_preview)

        preview_tab_layout.addWidget(preview_group)
        self.right_tabs.addTab(preview_tab, "DICOM Preview")

        analysis_tab = QWidget()
        analysis_tab_layout = QVBoxLayout(analysis_tab)

        analysis_group = QGroupBox("Analysis Preview")
        analysis_layout = QVBoxLayout(analysis_group)

        self.analysis_selector = QComboBox()
        self.analysis_selector.currentIndexChanged.connect(self.on_analysis_selection_changed)
        analysis_layout.addWidget(self.analysis_selector)

        self.analysis_preview = QLabel("No analysis preview available")
        self.analysis_preview.setAlignment(Qt.AlignCenter)
        self.analysis_preview.setMinimumHeight(420)
        self.analysis_preview.setStyleSheet("border: 1px solid gray; background-color: white;")
        analysis_layout.addWidget(self.analysis_preview)

        analysis_tab_layout.addWidget(analysis_group)
        self.right_tabs.addTab(analysis_tab, "Analysis Preview")

        results_tab = QWidget()
        results_tab_layout = QVBoxLayout(results_tab)

        results_plot_group = QGroupBox("Progression / Result Preview")
        results_plot_layout = QVBoxLayout(results_plot_group)

        self.results_plot_selector = QComboBox()
        self.results_plot_selector.currentIndexChanged.connect(self.on_results_plot_selection_changed)
        results_plot_layout.addWidget(self.results_plot_selector)

        self.results_plot_preview = QLabel("No progression plot available")
        self.results_plot_preview.setAlignment(Qt.AlignCenter)
        self.results_plot_preview.setMinimumHeight(620)
        self.results_plot_preview.setStyleSheet("border: 1px solid gray; background-color: white;")
        results_plot_layout.addWidget(self.results_plot_preview)

        results_tab_layout.addWidget(results_plot_group)
        self.right_tabs.addTab(results_tab, "Results")

    def _apply_initial_analysis_type(self) -> None:
        target_text = "Hip" if self.default_analysis_type == "hip" else "Hand"
        self.analysis_type_choice.blockSignals(True)
        self.analysis_type_choice.setCurrentText(target_text)
        self.analysis_type_choice.setEnabled(not self.lock_analysis_type)
        self.analysis_type_choice.blockSignals(False)

    def current_analysis_branch(self) -> str:
        return self.analysis_type_choice.currentText().strip().lower()

    def log(self, message: str) -> None:
        print("UI LOG:", message)
        self.status_box.append(message)

    def show_error(self, message: str) -> None:
        print("UI ERROR:", message)
        QMessageBox.critical(self, "Error", message)
        self.log(f"ERROR: {message}")

    def show_info(self, message: str) -> None:
        print("UI INFO:", message)
        QMessageBox.information(self, "Information", message)
        self.log(message)

    def require_patient_dir(self) -> Path:
        if self.current_patient_dir is None:
            raise WorkflowError("Select a patient first.")
        return self.current_patient_dir

    def require_study_dir(self) -> Path:
        if self.current_study_dir is None:
            raise WorkflowError("Select or assign a study first.")
        return self.current_study_dir

    def require_dicom_path(self) -> Path:
        if self.current_dicom_path is None:
            raise WorkflowError("Please select a DICOM image first.")
        return self.current_dicom_path

    def _is_hand_results(self, results: dict | None) -> bool:
        return bool(results) and results.get("analysis_type", "hand") == "hand"

    def _is_hip_results(self, results: dict | None) -> bool:
        return bool(results) and results.get("analysis_type") == "hip"

    def _results_dir(self) -> Path:
        study_dir = self.require_study_dir()
        if self.current_analysis_branch() == "hip":
            return study_dir / "hip_outputs"
        return study_dir / "outputs"

    def _partial_results_path(self) -> Path:
        return self._results_dir() / "results_partial.json"

    def _final_results_path(self) -> Path:
        study_dir = self.require_study_dir()
        if self.current_analysis_branch() == "hip":
            return study_dir / "hip_outputs" / "hip_results.json"
        return study_dir / "results.json"

    def _final_report_path(self) -> Path:
        if self.current_analysis_branch() == "hip":
            return self._results_dir() / "hip_report.txt"
        return self._results_dir() / "report.txt"

    def _final_csv_path(self) -> Path:
        if self.current_analysis_branch() == "hip":
            return self._results_dir() / "hip_results.csv"
        return self._results_dir() / "results.csv"

    # switch visible controls and instructions when the analysis type changes
    def on_analysis_type_changed(self) -> None:
        is_hand = self.current_analysis_branch() == "hand"
        self.hand_workflow_group.setVisible(is_hand)
        self.hip_workflow_group.setVisible(not is_hand)
        self.instructions_box.setPlainText(HAND_INSTRUCTIONS if is_hand else HIP_INSTRUCTIONS)
        self.last_analysis_type = None

        if self.current_study_dir is not None:
            self.latest_results = self.load_existing_results(silent=True)
        else:
            self.latest_results = None

        self.try_refresh_analysis_preview()
        self.try_refresh_results_plot_preview()
        self.update_results_box()

    # rescan the project folder and rebuild the patient list shown in the UI
    def refresh_patient_list(self) -> None:
        self.patient_list.clear()
        self.patient_choice.clear()

        existing_folders = []
        self.base_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.base_dir.iterdir()):
            if path.is_dir():
                existing_folders.append(path.name)
                self.patient_list.addItem(QListWidgetItem(path.name))
                self.patient_choice.addItem(path.name)

        if self.current_import_dicom_path is not None:
            try:
                ds = pydicom.dcmread(str(self.current_import_dicom_path), stop_before_pixels=True)
                suggested = get_patient_folder_name_from_dicom(ds)
                if suggested not in existing_folders:
                    self.patient_choice.insertItem(0, suggested)
                    self.patient_choice.setCurrentIndex(0)
            except Exception:
                pass

        self.log("Patient list refreshed.")

        if self.patient_choice.count() > 0 and self.patient_choice.currentIndex() < 0:
            self.patient_choice.setCurrentIndex(0)

    def get_first_dicom_in_study(self, study_dir: Path) -> Path | None:
        study_files_dir = study_dir / "study_files"
        if not study_files_dir.exists():
            return None

        for p in sorted(study_files_dir.iterdir()):
            if p.is_file():
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                    if hasattr(ds, "SOPInstanceUID"):
                        return p
                except Exception:
                    continue
        return None

    # load already-saved results for the current study if they exist
    # this lets the UI reopen a study without rerunning all analysis steps
    def load_existing_results(self, silent: bool = False) -> dict | None:
        study_dir = self.require_study_dir()

        if self.current_analysis_branch() == "hip":
            possible_files = [
                study_dir / "hip_outputs" / "results_partial.json",
                study_dir / "hip_outputs" / "hip_results.json",
            ]
        else:
            possible_files = [
                study_dir / "outputs" / "results_partial.json",
                study_dir / "results.json",
            ]

        for path in possible_files:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.latest_results = loaded
                if not silent:
                    self.log(f"Loaded existing results from {path}")
                return loaded

        return None

    def require_segmentation(self) -> dict:
        if self.latest_results and "segmentations" in self.latest_results:
            segmentation_data = self.latest_results["segmentations"]
        else:
            loaded_results = self.load_existing_results(silent=True)
            if not loaded_results or "segmentations" not in loaded_results:
                raise WorkflowError("Run segmentation first.")
            segmentation_data = loaded_results["segmentations"]

        if ("dicom_path" not in segmentation_data or not segmentation_data["dicom_path"]) and self.current_dicom_path is not None:
            segmentation_data["dicom_path"] = str(self.current_dicom_path)

        study_dir = self.current_study_dir
        if ("saved_masks" not in segmentation_data or not segmentation_data["saved_masks"]) and study_dir is not None:
            seg_dir = study_dir / "segmentations"
            if seg_dir.exists():
                preferred_order = [
                    "distal_phalanx.png",
                    "middle_phalanx.png",
                    "proximal_phalanx.png",
                    "metacarpal.png",
                ]

                rebuilt_paths = []
                for name in preferred_order:
                    p = seg_dir / name
                    if p.exists():
                        rebuilt_paths.append(str(p))

                for p in sorted(seg_dir.glob("*.png")):
                    if p.name == "sum_mask.png":
                        continue
                    if str(p) not in rebuilt_paths:
                        rebuilt_paths.append(str(p))

                segmentation_data["saved_masks"] = rebuilt_paths

        if ("mask_names" not in segmentation_data or not segmentation_data["mask_names"]) and segmentation_data.get("saved_masks"):
            inferred_names = []
            for p in segmentation_data["saved_masks"]:
                name = Path(p).stem.lower()
                if "distal" in name:
                    inferred_names.append("Distal Phalanx")
                elif "middle" in name:
                    inferred_names.append("Middle Phalanx")
                elif "proximal" in name:
                    inferred_names.append("Proximal Phalanx")
                elif "metacarpal" in name:
                    inferred_names.append("Metacarpal")
                else:
                    inferred_names.append(Path(p).stem.replace("_", " ").title())

            segmentation_data["mask_names"] = inferred_names

        if "dicom_path" not in segmentation_data or not segmentation_data["dicom_path"]:
            raise WorkflowError("No DICOM selected. Please select the study image again.")

        if "saved_masks" not in segmentation_data or not segmentation_data["saved_masks"]:
            raise WorkflowError("No saved masks found.")

        return segmentation_data

    def update_preview(self) -> None:
        if self.current_preview_array is None:
            self.image_preview.setText("No DICOM loaded")
            self.image_preview.setPixmap(QPixmap())
            return

        pixmap = numpy_to_qpixmap(self.current_preview_array)
        scaled = pixmap.scaled(
            max(100, self.image_preview.width() - 10),
            max(100, self.image_preview.height() - 10),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_preview.setPixmap(scaled)
        self.image_preview.setText("")

    def update_analysis_preview(self) -> None:
        if self.current_analysis_pixmap is None:
            self.analysis_preview.setText("No analysis preview available")
            self.analysis_preview.setPixmap(QPixmap())
            return

        scaled = self.current_analysis_pixmap.scaled(
            max(100, self.analysis_preview.width() - 10),
            max(100, self.analysis_preview.height() - 10),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.analysis_preview.setPixmap(scaled)
        self.analysis_preview.setText("")

    def update_results_plot_preview(self) -> None:
        if self.current_results_pixmap is None:
            default_text = (
                "No progression plot available"
                if self.current_analysis_branch() == "hand"
                else "No hip result preview available"
            )
            self.results_plot_preview.setText(default_text)
            self.results_plot_preview.setPixmap(QPixmap())
            return

        scaled = self.current_results_pixmap.scaled(
            max(100, self.results_plot_preview.width() - 10),
            max(100, self.results_plot_preview.height() - 10),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.results_plot_preview.setPixmap(scaled)
        self.results_plot_preview.setText("")

    def set_analysis_preview_from_path(self, image_path: Path) -> None:
        if not image_path.exists():
            self.current_analysis_pixmap = None
            self.update_analysis_preview()
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.current_analysis_pixmap = None
            self.update_analysis_preview()
            return

        self.current_analysis_pixmap = pixmap
        self.update_analysis_preview()

    def set_results_plot_preview_from_path(self, image_path: Path) -> None:
        if not image_path.exists():
            self.current_results_pixmap = None
            self.update_results_plot_preview()
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.current_results_pixmap = None
            self.update_results_plot_preview()
            return

        self.current_results_pixmap = pixmap
        self.update_results_plot_preview()

    def populate_analysis_selector(self, items: list[tuple[str, Path]]) -> None:
        self.analysis_selector.blockSignals(True)
        self.analysis_selector.clear()
        for label, path in items:
            self.analysis_selector.addItem(label, str(path))
        self.analysis_selector.blockSignals(False)

    def populate_results_plot_selector(self, items: list[tuple[str, Path]]) -> None:
        self.results_plot_selector.blockSignals(True)
        self.results_plot_selector.clear()
        for label, path in items:
            self.results_plot_selector.addItem(label, str(path))
        self.results_plot_selector.blockSignals(False)

    def on_analysis_selection_changed(self) -> None:
        path_str = self.analysis_selector.currentData()
        if path_str:
            self.set_analysis_preview_from_path(Path(path_str))

    def on_results_plot_selection_changed(self) -> None:
        path_str = self.results_plot_selector.currentData()
        if path_str:
            self.set_results_plot_preview_from_path(Path(path_str))

    def clear_analysis_preview(self) -> None:
        self.current_analysis_pixmap = None
        self.analysis_selector.blockSignals(True)
        self.analysis_selector.clear()
        self.analysis_selector.blockSignals(False)
        self.update_analysis_preview()

    def clear_results_plot_preview(self) -> None:
        self.current_results_pixmap = None
        self.results_plot_selector.blockSignals(True)
        self.results_plot_selector.clear()
        self.results_plot_selector.blockSignals(False)
        self.update_results_plot_preview()

    # refresh the Analysis Preview tab based on whichever workflow step was run most recently
    def try_refresh_analysis_preview(self) -> None:
        if self.current_study_dir is None:
            self.clear_analysis_preview()
            return

        items: list[tuple[str, Path]] = []

        if self.current_analysis_branch() == "hip":
            hip_preview = self.current_study_dir / "hip_outputs" / "hip_analysis_preview.png"
            if hip_preview.exists():
                items.append(("Hip Analysis", hip_preview))
                self.populate_analysis_selector(items)
                self.analysis_selector.setCurrentIndex(0)
                self.set_analysis_preview_from_path(hip_preview)
            else:
                self.clear_analysis_preview()
            return

        outputs_dir = self.current_study_dir / "outputs"
        segmentation_preview = self.current_study_dir / "segmentations" / "sum_mask.png"
        curvature_preview = outputs_dir / "curvature_visual.png"
        ratio_previews = sorted(outputs_dir.glob("ratio_visual_*.png"))

        if segmentation_preview.exists():
            items.append(("Segmentation", segmentation_preview))

        for p in ratio_previews:
            label = p.stem.replace("ratio_visual_", "").replace("_", " ").title()
            items.append((f"Ratio: {label}", p))

        if curvature_preview.exists():
            items.append(("Curvature", curvature_preview))

        if not items:
            self.clear_analysis_preview()
            return

        self.populate_analysis_selector(items)
        selected_index = 0

        if self.last_analysis_type == "ratio":
            for i, (label, _) in enumerate(items):
                if label.startswith("Ratio:"):
                    selected_index = i
                    break
        elif self.last_analysis_type == "curvature":
            for i, (label, _) in enumerate(items):
                if label == "Curvature":
                    selected_index = i
                    break
        elif self.last_analysis_type == "segmentation":
            for i, (label, _) in enumerate(items):
                if label == "Segmentation":
                    selected_index = i
                    break

        self.analysis_selector.setCurrentIndex(selected_index)
        path_str = self.analysis_selector.currentData()
        if path_str:
            self.set_analysis_preview_from_path(Path(path_str))
        else:
            self.clear_analysis_preview()

    # refresh the Results tab by scanning the output folder for saved progression plots
    def try_refresh_results_plot_preview(self) -> None:
        if self.current_patient_dir is None:
            self.clear_results_plot_preview()
            return

        if self.current_analysis_branch() == "hip":
            items: list[tuple[str, Path]] = []

            plot_dir = self.current_patient_dir / "analysis_data" / "hip_plots"
            if plot_dir.exists():
                for p in sorted(plot_dir.glob("hip_plot_*.png")):
                    label = p.stem.replace("hip_plot_", "").replace("_", " ").title()
                    items.append((label, p))

            if self.current_study_dir is not None:
                hip_preview = self.current_study_dir / "hip_outputs" / "hip_analysis_preview.png"
                if hip_preview.exists():
                    items.append(("Current Study Preview", hip_preview))

            if not items:
                self.clear_results_plot_preview()
                return

            self.populate_results_plot_selector(items)
            self.results_plot_selector.setCurrentIndex(0)
            path_str = self.results_plot_selector.currentData()
            if path_str:
                self.set_results_plot_preview_from_path(Path(path_str))
            else:
                self.clear_results_plot_preview()
            return

        plot_dir = self.current_patient_dir / "analysis_data" / "plots"
        items: list[tuple[str, Path]] = []

        if plot_dir.exists():
            for p in sorted(plot_dir.glob("plot_*.png")):
                label = p.stem.replace("plot_", "").replace("_", " ").title()
                items.append((label, p))

        if not items:
            self.clear_results_plot_preview()
            return

        self.populate_results_plot_selector(items)
        self.results_plot_selector.setCurrentIndex(0)
        path_str = self.results_plot_selector.currentData()
        if path_str:
            self.set_results_plot_preview_from_path(Path(path_str))
        else:
            self.clear_results_plot_preview()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_preview()
        self.update_analysis_preview()
        self.update_results_plot_preview()

    # show the latest results JSON in the UI so the user can quickly inspect saved values
    def update_results_box(self) -> None:
        return

    def clear_study_list(self) -> None:
        self.study_list.clear()
        self.current_opened_study_files = []

    def populate_import_candidate_list(self, dicom_files: list[Path]) -> None:
        self.clear_study_list()
        self.current_list_mode = "import_candidates"
        self.current_opened_study_files = dicom_files

        for i, dicom_file in enumerate(dicom_files, start=1):
            label = dicom_file.name
            try:
                ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                instance = str(getattr(ds, "InstanceNumber", i))
                year = get_study_year_from_dicom(ds)
                series_desc = str(getattr(ds, "SeriesDescription", "")).strip()
                image_type = getattr(ds, "ImageType", "")
                if isinstance(image_type, (list, tuple)):
                    image_type = " / ".join(str(x) for x in image_type if x)
                else:
                    image_type = str(image_type).strip()

                label = f"{year} | {instance} | {dicom_file.name}"
                if series_desc:
                    label += f" | {series_desc}"
                elif image_type:
                    label += f" | {image_type}"
            except Exception:
                pass

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, {"mode": "import", "path": str(dicom_file)})
            self.study_list.addItem(item)

        if dicom_files:
            self.study_list.setCurrentRow(0)
            self.load_import_dicom_path(dicom_files[0])

    def populate_saved_study_list(self, study_dirs: list[Path]) -> None:
        self.clear_study_list()
        self.current_list_mode = "saved_studies"

        for study_dir in study_dirs:
            dicom_path = self.get_first_dicom_in_study(study_dir)
            if dicom_path is None:
                continue

            label = study_dir.name
            try:
                ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
                label = get_study_display_from_dicom(ds, fallback_filename=dicom_path.name)
            except Exception:
                pass

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, {"mode": "saved", "study_dir": str(study_dir)})
            self.study_list.addItem(item)

        if self.study_list.count() > 0:
            self.study_list.setCurrentRow(0)
            first_item = self.study_list.item(0)
            if first_item is not None:
                self.on_study_list_item_clicked(first_item)

    def group_saved_studies_by_year(self, patient_dir: Path) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = {}
        studies_root = patient_dir / "studies"

        if not studies_root.exists():
            return grouped

        for study_dir in sorted(studies_root.iterdir()):
            if not study_dir.is_dir():
                continue

            dicom_path = self.get_first_dicom_in_study(study_dir)
            year = "Unknown"
            if dicom_path is not None:
                try:
                    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
                    year = get_study_year_from_dicom(ds)
                except Exception:
                    year = "Unknown"

            grouped.setdefault(year, []).append(study_dir)

        def sort_key(year_str: str):
            return (0, int(year_str)) if year_str.isdigit() else (1, year_str)

        sorted_years = sorted(grouped.keys(), key=sort_key, reverse=True)
        return {year: sorted(grouped[year]) for year in sorted_years}

    def load_import_dicom_path(self, dicom_path: Path) -> None:
        ds, preview_array = load_dicom_for_preview(dicom_path)
        self.current_import_dicom_path = dicom_path
        self.current_dicom_path = dicom_path
        self.current_preview_array = preview_array
        self.current_patient_display = get_patient_display_from_dicom(ds)
        self.current_study_display = get_study_display_from_dicom(ds, fallback_filename=dicom_path.name)

        self.image_path_label.setText(str(dicom_path))
        self.detected_patient_label.setText(self.current_patient_display)
        self.study_assignment_label.setText(f"Import candidate selected:\n{self.current_study_display}")
        self.update_preview()
        self.refresh_patient_list()
        self.log(f"Import DICOM opened: {dicom_path}")

    def load_saved_study(self, study_dir: Path) -> None:
        dicom_path = self.get_first_dicom_in_study(study_dir)
        if dicom_path is None:
            raise WorkflowError("No readable DICOM found in this saved study.")

        ds, preview_array = load_dicom_for_preview(dicom_path)
        self.current_study_dir = study_dir
        self.current_dicom_path = dicom_path
        self.current_preview_array = preview_array
        self.current_patient_display = get_patient_display_from_dicom(ds)
        self.current_study_display = get_study_display_from_dicom(ds, fallback_filename=dicom_path.name)

        self.image_path_label.setText(str(dicom_path))
        self.detected_patient_label.setText(self.current_patient_display)
        self.study_assignment_label.setText(f"Active saved study: {study_dir.name}\n{self.current_study_display}")

        self.update_preview()
        self.try_refresh_analysis_preview()
        self.try_refresh_results_plot_preview()
        self.log(f"Saved study loaded: {study_dir}")

        try:
            loaded_results = self.load_existing_results(silent=True)
            self.latest_results = loaded_results
            self.update_results_box()
        except Exception as e:
            self.log(f"Could not load saved results: {e}")

    # open a single DICOM file from disk
    def select_dicom(self) -> None:
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open DICOM",
                "",
                "DICOM Files (*.dcm *.dicom);;All Files (*)",
            )
            if not file_path:
                return
            self.populate_import_candidate_list([Path(file_path)])
        except Exception as e:
            self.show_error(str(e))

    # open a folder and list all valid DICOM files found inside it
    def open_dicom_folder(self) -> None:
        try:
            folder = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
            if not folder:
                return

            folder_path = Path(folder)
            dicom_files = []

            for file_path in sorted(folder_path.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".dcm", ".dicom", ""}:
                    continue

                try:
                    ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                    if hasattr(ds, "SOPInstanceUID"):
                        dicom_files.append(file_path)
                except Exception:
                    continue

            if not dicom_files:
                raise WorkflowError("No readable DICOM files were found in that folder.")

            def sort_key(path: Path):
                try:
                    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                    study_year = get_study_year_from_dicom(ds)
                    series_num = int(getattr(ds, "SeriesNumber", 0))
                    instance_num = int(getattr(ds, "InstanceNumber", 0))
                    return (study_year, series_num, instance_num, path.name)
                except Exception:
                    return ("Unknown", 0, 0, path.name)

            dicom_files.sort(key=sort_key)
            self.populate_import_candidate_list(dicom_files)
            self.log(f"Loaded DICOM folder with {len(dicom_files)} files.")
        except Exception as e:
            self.show_error(str(e))

    # load the clicked study or image entry into the preview area
    def on_study_list_item_clicked(self, item: QListWidgetItem) -> None:
        try:
            payload = item.data(Qt.UserRole) or {}
            mode = payload.get("mode")

            if mode == "import":
                self.load_import_dicom_path(Path(payload["path"]))
            elif mode == "saved":
                self.load_saved_study(Path(payload["study_dir"]))
            else:
                raise WorkflowError("Unknown list item type.")
        except Exception as e:
            self.show_error(str(e))

    # create or reuse the patient/study folders and copy the selected DICOM into the project structure
    def assign_patient_and_study(self) -> None:
        try:
            import_dicom_path = self.current_import_dicom_path
            if import_dicom_path is None:
                raise WorkflowError("Open and select an image from disk first.")

            selected_patient_folder = self.patient_choice.currentText().strip()
            if not selected_patient_folder:
                raise WorkflowError("No patient folder selected.")

            ds = pydicom.dcmread(str(import_dicom_path), stop_before_pixels=True)

            patient_dir = ensure_patient_structure(selected_patient_folder, base_dir=self.base_dir)
            study_folder = get_study_folder_name_from_dicom(ds, import_dicom_path)
            study_dir = ensure_study_structure(selected_patient_folder, study_folder, base_dir=self.base_dir)
            copied_path = copy_dicom_to_study(import_dicom_path, study_dir)

            self.current_patient_dir = patient_dir
            self.current_study_dir = study_dir
            self.current_dicom_path = copied_path
            self.current_patient_display = get_patient_display_from_dicom(ds)
            self.current_study_display = get_study_display_from_dicom(ds, fallback_filename=copied_path.name)

            self.selected_patient_label.setText(patient_dir.name)
            self.study_assignment_label.setText(f"Assigned study: {study_dir.name}\n{self.current_study_display}")
            self.image_path_label.setText(str(copied_path))

            self.log(f"Using patient folder: {selected_patient_folder}")
            self.log(f"Using study folder: {study_folder}")
            self.show_info("Selected image was imported into the patient and saved as its own study folder.")

            self.latest_results = None
            self.update_results_box()
            self.refresh_patient_list()
            self.reload_current_patient_studies(select_study_dir=study_dir)
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
        except Exception as e:
            self.show_error(str(e))

    # reopen an existing patient so earlier studies and results can be browsed again
    def load_selected_patient(self, item: QListWidgetItem) -> None:
        folder_name = item.text()
        self.current_patient_dir = ensure_patient_structure(folder_name, base_dir=self.base_dir)
        self.selected_patient_label.setText(folder_name)
        self.patient_choice.setCurrentText(folder_name)
        self.log(f"Selected existing patient folder: {folder_name}")
        self.reload_current_patient_studies(select_study_dir=None)

    def reload_current_patient_studies(self, select_study_dir: Path | None) -> None:
        patient_dir = self.require_patient_dir()
        self.current_saved_studies_by_year = self.group_saved_studies_by_year(patient_dir)

        self.year_choice.blockSignals(True)
        self.year_choice.clear()
        years = list(self.current_saved_studies_by_year.keys())
        for year in years:
            self.year_choice.addItem(year)
        self.year_choice.blockSignals(False)

        if not years:
            self.clear_study_list()
            self.current_study_dir = None
            self.latest_results = None
            self.update_results_box()
            self.clear_analysis_preview()
            self.clear_results_plot_preview()
            self.study_assignment_label.setText("No saved studies for this patient yet")
            return

        target_year = None
        if select_study_dir is not None:
            for year, study_dirs in self.current_saved_studies_by_year.items():
                if any(study_dir == select_study_dir for study_dir in study_dirs):
                    target_year = year
                    break

        if target_year is None:
            target_year = years[0]

        idx = self.year_choice.findText(target_year)
        if idx >= 0:
            self.year_choice.setCurrentIndex(idx)

        self.populate_saved_study_list(self.current_saved_studies_by_year.get(target_year, []))
        self.try_refresh_results_plot_preview()

        if select_study_dir is not None:
            for row in range(self.study_list.count()):
                item = self.study_list.item(row)
                payload = item.data(Qt.UserRole) or {}
                if payload.get("mode") == "saved" and Path(payload.get("study_dir", "")) == select_study_dir:
                    self.study_list.setCurrentRow(row)
                    self.on_study_list_item_clicked(item)
                    break

    def on_year_changed(self) -> None:
        if self.current_patient_dir is None:
            return

        selected_year = self.year_choice.currentText().strip()
        if not selected_year:
            self.clear_study_list()
            return

        study_dirs = self.current_saved_studies_by_year.get(selected_year, [])
        self.populate_saved_study_list(study_dirs)
        self.log(f"Loaded {len(study_dirs)} saved studies for year {selected_year}")

    def _current_dicom_metadata(self, dicom_path: Path) -> dict:
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        return {
            "study_date": get_study_date_from_dicom(ds),
            "age_months": get_patient_age_months_from_dicom(ds),
            "age_years": get_patient_age_years_from_dicom(ds),
            "bone_age_months": get_bone_age_months_from_dicom(ds),
        }

    def _build_hand_results_with_current_age(
        self,
        patient_dir: Path,
        study_dir: Path,
        dicom_path: Path,
        segmentation_data: dict,
        ratio_results: list[dict] | None = None,
        curvature_results: dict | None = None,
    ) -> dict:
        md = self._current_dicom_metadata(dicom_path)
        return build_hand_results(
            patient_dir=patient_dir,
            patient_display=self.current_patient_display,
            study_dir=study_dir,
            study_display=self.current_study_display,
            dicom_path=dicom_path,
            segmentation_data=segmentation_data,
            ratio_results=ratio_results,
            curvature_results=curvature_results,
            group_label="Current patient",
            study_date=md["study_date"],
            age_months=md["age_months"],
            age_years=md["age_years"],
            bone_age_months=md["bone_age_months"],
        )

    def _build_hip_results_with_current_age(
        self,
        patient_dir: Path,
        study_dir: Path,
        dicom_path: Path,
        hip_analysis: dict,
    ) -> dict:
        md = self._current_dicom_metadata(dicom_path)
        return build_hip_results(
            patient_dir=patient_dir,
            patient_display=self.current_patient_display,
            study_dir=study_dir,
            study_display=self.current_study_display,
            dicom_path=dicom_path,
            hip_data=hip_analysis,
            group_label="Current patient",
            study_date=md["study_date"],
            age_months=md["age_months"],
            age_years=md["age_years"],
            bone_age_months=md["bone_age_months"],
        )

    def _save_study_and_patient_level_outputs(self) -> None:
        partial_path = self._partial_results_path()
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(partial_path, self.latest_results)

        if self.current_analysis_branch() == "hip":
            save_hip_results_csv(self._results_dir() / "hip_results.csv", self.latest_results)
        else:
            save_hand_results_csv(self._results_dir() / "results.csv", self.latest_results)
            patient_dir = self.require_patient_dir()
            update_hand_patient_analysis_files(patient_dir, self.latest_results)

    # step 1 for the hand workflow
    # run the interactive lasso segmentation and save partial outputs immediately
    def run_segmentation_step(self) -> None:
        try:
            if self.current_analysis_branch() != "hand":
                raise WorkflowError("Switch to Hand analysis to run segmentation.")

            patient_dir = self.require_patient_dir()
            study_dir = self.require_study_dir()
            dicom_path = self.require_dicom_path()

            # the saved segmentation dictionary is reused later for ratios and curvature
            segmentation_data = run_lasso_segmentation(
                dicom_path=dicom_path,
                save_dir=study_dir / "segmentations",
            )

            self.latest_results = self._build_hand_results_with_current_age(
                patient_dir=patient_dir,
                study_dir=study_dir,
                dicom_path=dicom_path,
                segmentation_data=segmentation_data,
            )

            self._save_study_and_patient_level_outputs()
            self.last_analysis_type = "segmentation"
            self.update_results_box()
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(1)
            self.show_info("Segmentation finished and saved.")
        except Exception as e:
            self.show_error(str(e))

    # step 2 for the hand workflow
    # calculate finger length and width measurements from the saved masks
    def run_ratio_step(self) -> None:
        try:
            if self.current_analysis_branch() != "hand":
                raise WorkflowError("Switch to Hand analysis to run finger ratios.")

            patient_dir = self.require_patient_dir()
            study_dir = self.require_study_dir()
            segmentation_data = self.require_segmentation()
            ratio_results = calculate_finger_ratios(segmentation_data)

            if self.latest_results is None or not self._is_hand_results(self.latest_results):
                self.latest_results = self.load_existing_results(silent=True)

            if self.latest_results is None or not self._is_hand_results(self.latest_results):
                self.latest_results = self._build_hand_results_with_current_age(
                    patient_dir=patient_dir,
                    study_dir=study_dir,
                    dicom_path=self.require_dicom_path(),
                    segmentation_data=segmentation_data,
                )

            self.latest_results["segmentations"] = segmentation_data
            self.latest_results["finger_ratios"] = ratio_results
            self._save_study_and_patient_level_outputs()

            self.last_analysis_type = "ratio"
            self.update_results_box()
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(1)
            self.show_info("Finger ratio measurements finished.")
        except Exception as e:
            self.show_error(str(e))

    # step 3 for the hand workflow
    # calculate joint angles from the segmented bone masks
    def run_curvature_step(self) -> None:
        try:
            if self.current_analysis_branch() != "hand":
                raise WorkflowError("Switch to Hand analysis to run curvature.")

            patient_dir = self.require_patient_dir()
            study_dir = self.require_study_dir()
            segmentation_data = self.require_segmentation()
            curvature_results = calculate_curvature(segmentation_data)

            if self.latest_results is None or not self._is_hand_results(self.latest_results):
                self.latest_results = self.load_existing_results(silent=True)

            if self.latest_results is None or not self._is_hand_results(self.latest_results):
                self.latest_results = self._build_hand_results_with_current_age(
                    patient_dir=patient_dir,
                    study_dir=study_dir,
                    dicom_path=self.require_dicom_path(),
                    segmentation_data=segmentation_data,
                )

            self.latest_results["segmentations"] = segmentation_data
            self.latest_results["curvature"] = curvature_results
            self._save_study_and_patient_level_outputs()

            self.last_analysis_type = "curvature"
            self.update_results_box()
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(1)
            self.show_info("Curvature analysis finished.")
        except Exception as e:
            self.show_error(str(e))

    # convenience button that runs segmentation, ratio and curvature in sequence
    def run_full_workflow(self) -> None:
        try:
            if self.current_analysis_branch() != "hand":
                raise WorkflowError("Switch to Hand analysis to run the full hand workflow.")

            patient_dir = self.require_patient_dir()
            study_dir = self.require_study_dir()
            dicom_path = self.require_dicom_path()

            segmentation_data = run_lasso_segmentation(
                dicom_path=dicom_path,
                save_dir=study_dir / "segmentations",
            )
            ratio_results = calculate_finger_ratios(segmentation_data)
            curvature_results = calculate_curvature(segmentation_data)

            self.latest_results = self._build_hand_results_with_current_age(
                patient_dir=patient_dir,
                study_dir=study_dir,
                dicom_path=dicom_path,
                segmentation_data=segmentation_data,
                ratio_results=ratio_results,
                curvature_results=curvature_results,
            )

            self._save_study_and_patient_level_outputs()

            final_results_path = self._final_results_path()
            final_results_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(final_results_path, self.latest_results)
            save_report(self._final_report_path(), build_hand_text_report(self.latest_results))
            save_hand_results_csv(self._final_csv_path(), self.latest_results)

            self.last_analysis_type = "curvature"
            self.update_results_box()
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(1)
            self.show_info("Full hand workflow finished and saved.")
        except Exception as e:
            self.show_error(str(e))

    # main hip workflow step
    # open the semi-automatic acetabulum tool and save the returned measurements
    def run_hip_analysis_step(self) -> None:
        try:
            if self.current_analysis_branch() != "hip":
                raise WorkflowError("Switch to Hip analysis to run the hip tool.")

            patient_dir = self.require_patient_dir()
            study_dir = self.require_study_dir()
            dicom_path = self.require_dicom_path()

            hip_output_dir = study_dir / "hip_outputs"
            hip_output_dir.mkdir(parents=True, exist_ok=True)

            md = self._current_dicom_metadata(dicom_path)

            hip_analysis = run_hip_acetabulum_analysis(
                dicom_path=dicom_path,
                output_dir=hip_output_dir,
                patient_id=patient_dir.name,
                scan_date=md["study_date"],
                image_label=dicom_path.name,
            )

            if hip_analysis is None:
                self.log("Hip analysis cancelled before completion.")
                return

            self.latest_results = self._build_hip_results_with_current_age(
                patient_dir=patient_dir,
                study_dir=study_dir,
                dicom_path=dicom_path,
                hip_analysis=hip_analysis,
            )

            self._save_study_and_patient_level_outputs()

            final_results_path = self._final_results_path()
            final_results_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(final_results_path, self.latest_results)
            save_report(self._final_report_path(), build_hip_text_report(self.latest_results))
            save_hip_results_csv(self._final_csv_path(), self.latest_results)

            self.last_analysis_type = "hip"
            self.update_results_box()
            self.try_refresh_analysis_preview()
            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(1)
            self.show_info("Hip acetabulum analysis finished and saved.")

        except WorkflowError as e:
            self.show_error(str(e))
        except Exception as e:
            self.show_error(f"Hip analysis could not be completed: {e}")

    # write the latest in-memory results to the final JSON, CSV and text report files
    def save_final_report(self) -> None:
        try:
            patient_dir = self.require_patient_dir()

            if not self.latest_results:
                loaded = self.load_existing_results(silent=True)
                if not loaded:
                    raise WorkflowError("No results available to save.")
                self.latest_results = loaded

            final_results_path = self._final_results_path()
            final_results_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(final_results_path, self.latest_results)

            if self.current_analysis_branch() == "hip":
                save_report(self._final_report_path(), build_hip_text_report(self.latest_results))
                save_hip_results_csv(self._final_csv_path(), self.latest_results)
            else:
                save_report(self._final_report_path(), build_hand_text_report(self.latest_results))
                save_hand_results_csv(self._final_csv_path(), self.latest_results)
                update_hand_patient_analysis_files(patient_dir, self.latest_results)

            self.latest_results = make_results_json_safe(self.latest_results)
            self.try_refresh_results_plot_preview()
            self.show_info("Final report saved.")
        except Exception as e:
            self.show_error(str(e))

    # generate patient-level progression plots from the accumulated saved study results
    def generate_progression_plots_step(self) -> None:
        try:
            patient_dir = self.require_patient_dir()

            if self.current_analysis_branch() == "hand":
                from progression_plots import generate_progression_plots_for_patient
                plot_paths = generate_progression_plots_for_patient(patient_dir)

            elif self.current_analysis_branch() == "hip":
                from hip_progression_plots import generate_hip_progression_plots_for_patient
                plot_paths = generate_hip_progression_plots_for_patient(patient_dir)

            else:
                raise WorkflowError("Unknown analysis branch.")

            if not plot_paths:
                raise WorkflowError("No progression plots could be generated.")

            self.try_refresh_results_plot_preview()
            self.right_tabs.setCurrentIndex(2)
            self.show_info(f"Generated {len(plot_paths)} progression plots.")
        except Exception as e:
            self.show_error(str(e))
