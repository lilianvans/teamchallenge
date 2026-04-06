import sys
from PyQt5.QtWidgets import QApplication
from ui_launcher import LauncherWindow


# start the Qt application and open the launcher window
def main():
    app = QApplication(sys.argv)

    window = LauncherWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
