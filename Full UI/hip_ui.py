from pathlib import Path
from workflow_ui import BaseWorkflowWindow

HIP_BASE_DIR = Path("data/hip_patients")


# small wrapper window that locks the shared workflow UI to one analysis branch
class HipWorkflowWindow(BaseWorkflowWindow):
    def __init__(self) -> None:
        super().__init__(
            base_dir=HIP_BASE_DIR,
            analysis_type="hip",
            lock_analysis_type=True,
        )
        self.setWindowTitle("Hip Analysis Workflow")
