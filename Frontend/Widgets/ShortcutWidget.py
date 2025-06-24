from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QMessageBox, QDialog, QVBoxLayout, QTabWidget, QLabel, QFileDialog
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton
from Frontend.Widgets.Settings import Settings
from Frontend.Widgets.DirectoryWidget import DirectoryWidget
from Frontend.Handlers.ExportHandler import ExportHandler

class ShortcutWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setGeometry(100, 100, 300, 150)

        layout = QGridLayout()
        layout.setSpacing(1) 
        layout.setContentsMargins(1, 1, 1, 1)
        
        self.OpenABF = QPushButton("Open File")
        self.OpenABF.setShortcut("Ctrl+Shift+O")
        self.OpenABF.clicked.connect(self.OpenFile)

        self.SaveSesh = QPushButton("Save Session")
        self.SaveSesh.clicked.connect(self.SaveSessionFunction)

        self.LoadSesh = QPushButton("Load Session")
        self.LoadSesh.clicked.connect(self.LoadSessionFunction)

        self.Settings = QPushButton("Settings")
        self.Settings.setShortcut("Ctrl+Shift+S")
        self.Settings.clicked.connect(self.SettingsWindow)

        self.ExportResults = QPushButton("Export Results")
        self.ExportResults.clicked.connect(self.ExportResultsFunction)

        self.Help = QPushButton("Help")
        self.Help.setShortcut("Ctrl+Shift+H")
        self.Help.clicked.connect(self.HelpWindow)

        layout.addWidget(self.OpenABF, 0, 0)
        layout.addWidget(self.SaveSesh, 1, 0)
        layout.addWidget(self.LoadSesh, 2, 0)
        layout.addWidget(self.ExportResults, 3, 0)
        layout.addWidget(self.Settings, 4, 0)
        layout.addWidget(self.Help, 5, 0)

        self.setMinimumSize(357, 150)
        
        self.setLayout(layout)
        
    def handle_export_results(self):
        if not self.parent:
            return
    
        results_box = self.parent.TabbedFunction.ResultsBox
        plot_widget = self.parent.PlotArea.channel0_Plot  # pyqtgraph main plot
    
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        dialog.setMinimumSize(300, 150)
    
        layout = QVBoxLayout()
        tabs = QTabWidget()
    
        # Tab 1: Export Results
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_layout.addWidget(QLabel("Export only the Results tab."))
        export_results_btn = QPushButton("Export Results")
        export_results_btn.clicked.connect(lambda: (ExportHandler.export_results_to_csv(results_box), dialog.accept()))
        results_layout.addWidget(export_results_btn)
        results_tab.setLayout(results_layout)
    
        # Tab 2: Export Graph
        graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(QLabel("Export only the current graph view."))
        export_graph_btn = QPushButton("Export Graph")
        export_graph_btn.clicked.connect(lambda: (ExportHandler.export_plot_to_png(plot_widget), dialog.accept()))
        graph_layout.addWidget(export_graph_btn)
        graph_tab.setLayout(graph_layout)
    
        # Add tabs to dialog
        tabs.addTab(results_tab, "Results")
        tabs.addTab(graph_tab, "Graph")
    
        layout.addWidget(tabs)
        dialog.setLayout(layout)
        dialog.exec_()
            


    def ExportResultsFunction(self):
        if self.parent:
            handler = ExportHandler(
                result_widget=self.parent.TabbedFunction.ResultsBox,
                plot_area=self.parent.PlotArea,
                analysis_type=self.parent.TypeSelector.tabs.tabText(
                    self.parent.TypeSelector.tabs.currentIndex()
                ),
                parent=self
            )
            handler.exec_()

    def OpenFile(self):
        if self.parent:
            self.parent.browse_file()

    def SaveSessionFunction(self):
        if self.parent:
            self.parent.save_session()

    def LoadSessionFunction(self):
        if self.parent:
            self.parent.load_session()

    def RestartFunction(self):
        if self.parent:
            self.parent.restart_app()

    def SettingsWindow(self):
        if self.parent:
            self.parent.SettingWindow.show()

    def HelpWindow(self):
        if self.parent:
            self.parent.HelpBook.show()
