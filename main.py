import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from PyQt5.QtWidgets import QApplication
import traceback, sys
from Frontend.Widgets.AnalysisSelectorWidget import AnalysisSelector #For testing individual Widgets
from Frontend.Widgets.DirectoryWidget import DirectoryWidget #For testing individual Widgets
from Frontend.Widgets.IxDPlotAreaWidget import IxDPlotArea #For testing individual Widgets
from Frontend.Widgets.ShortcutWidget import ShortcutWidget #For testing individual Widgets
from Frontend.Widgets.Tabbedwidget import TabbedWindow #For testing individual Widgets
from Frontend.main import GridStyleApplication

import matplotlib
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = GridStyleApplication()
        window.show()
        sys.exit(app.exec_())
        
    except Exception:
        traceback.print_exc()
        # print(f"Error occured:")
        sys.exit(1)
