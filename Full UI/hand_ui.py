from pathlib import Path
from workflow_ui import BaseWorkflowWindow

HAND_BASE_DIR = Path("data/patients")


# small wrapper window that locks the shared workflow UI to one analysis branch
class HandWorkflowWindow(BaseWorkflowWindow):
    def __init__(self) -> None:
        super().__init__(
            base_dir=HAND_BASE_DIR,
            analysis_type="hand",
            lock_analysis_type=True,
        )
        self.setWindowTitle("Hand Analysis Workflow")
