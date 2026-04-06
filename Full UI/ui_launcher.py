from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel

from hand_ui import HandWorkflowWindow
from hip_ui import HipWorkflowWindow


# simple launcher window so the user can choose between hand and hip workflows
class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Select Analysis")
        self.resize(300, 180)

        layout = QVBoxLayout()

        title = QLabel("Choose workflow")
        layout.addWidget(title)

        self.hand_btn = QPushButton("Hand Analysis")
        self.hip_btn = QPushButton("Hip Analysis")

        self.hand_btn.clicked.connect(self.open_hand)
        self.hip_btn.clicked.connect(self.open_hip)

        layout.addWidget(self.hand_btn)
        layout.addWidget(self.hip_btn)

        self.setLayout(layout)

        self.hand_window = None
        self.hip_window = None

    # open a hand workflow window
    def open_hand(self):
        self.hand_window = HandWorkflowWindow()
        self.hand_window.show()

    # open a hip workflow window
    def open_hip(self):
        self.hip_window = HipWorkflowWindow()
        self.hip_window.show()
