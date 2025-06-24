import sys
import os
import subprocess
import pyabf
import csv
import joblib
import numpy as np
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QFileDialog , QAction, QStatusBar, QVBoxLayout, 
    QApplication, QMessageBox, QDockWidget, QSizePolicy, QLabel
)
from PyQt5.QtCore import Qt

#Handler Class
from Frontend.Handlers.ErrorHandler import ErrorHandler
from Frontend.Handlers.MessageHandler import MessageHandler
from Frontend.Handlers.ResultsHandler import ResultHandler

#Imported Widgets
from Frontend.Widgets.HelpBook import HelpBook
from Frontend.Widgets.AnalysisSelectorWidget import AnalysisSelector
from Frontend.Widgets.Settings import Settings

from Frontend.Widgets.Tabbedwidget import TabbedWindow
from Frontend.Widgets.DirectoryWidget import DirectoryWidget
from Frontend.Widgets.IxDPlotAreaWidget import IxDPlotArea 
from Frontend.Widgets.ShortcutWidget import ShortcutWidget

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Threads.Threadcontroller import NeuroThreadController

class GridStyleApplication(QMainWindow):
    #Constructor for the Main window class
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neurolyzer")
        screenMeasurements = QApplication.primaryScreen().availableGeometry()
        # self.setGeometry(screenMeasurements)
        windowWidth = int(screenMeasurements.width() * 0.75)
        windowHeight = int(screenMeasurements.height() * 0.75)
        self.resize(windowWidth, windowHeight)
        self.directoryPath = None
        
        #Controller being made before UI
        self.controller = NeuroThreadController()
        
        self.segment = {
            "Baseline": [],
            "HighK": [],
            "Recovery": []
        }
        
        self.heatmap = {
            "labels": [],
            "f": [],
            "t": [],
            "Sxx": [],
            "freq": [],
            "time": [],
            "power": [],
            "Spec_min": [],
            "Spec_max": []
        }
        
        self.spectrum = {
            "labels": [],
            "freqs": [],
            "mean_psd": [],
            "sem_psd": []
        }
        
        self.results = {
            "Analysis Type": [],
            "Value": []
        }
        
        self.zap = {
            "Frequency": [],
            "Impedance": []
        }
        
        self.clearance = {
            "peak": [],
            "t0": [],
            "v_baseline": [],
            "v_peak": [],
            "delta_v": [],
            "k_baseline": [],
            "k_peak": [],
            "delta_k": [],
            "tau": [],
            "lambda": [],
            "r2": [],
            "amps": [],
            "C": [],
            "decay_time": [],
            "slice_x": [],
            "slice_y": [],
            "fit_x": [],
            "fit_y": [],
            "k90": [],
            "k10": [],
            "full_x": [],
            "full_y": [],
            "t_win": [],
            "y_win": [],
            "n_data": [],
            "slice_t0": [],
            "slice_dt": [],
            "seg_x": [],
            "seg_y": [],
            "fitted": [],
            "v90": [],
            "v10": [],
            "tau_s_str": [],
            "lam_s_str": [],
            "delay_time": [],
            "A_str": []
        }
        
        #ToolBar Menu
        self.setup_menu()
        self.setup_status_bar()
        self.setup_ui()

        # Restore settings after UI setup
        self.restore_user_settings()
        self.selected_file = None
        
        self.resultexport = {
            "Analysis Type": [],
            "Metric": [],
            "Value": []
        }
        
        #Sets up variables after setting up the UI
        self.selected_file = None
        self.analysis_buttons = []

    #Top Toolbar Menu Functions 
    def setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        
        #Funtionality for ABF file Loader Button 
        open_action = QAction("Open ABF File", self)
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)

        #Functionality for Export as CSV button
        export_csv_action = QAction("Export ABF data (csv only)", self)
        export_csv_action.triggered.connect(self.export_abf_to_csv)
        file_menu.addAction(export_csv_action)

        #Functionality for Login Feature
        login_action = QAction("Login to Account", self)
        login_action.triggered.connect(lambda: self.log_action("Login to Account"))
        file_menu.addAction(login_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_app)
        file_menu.addAction(quit_action)

        # Tools Menu
        tools_menu = menubar.addMenu("Tools")

        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+Shift+S")
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
    
        clear_console_action = QAction("Clear Console", self)
        clear_console_action.setShortcut("Ctrl+K")
        clear_console_action.triggered.connect(self.clear_console)
        tools_menu.addAction(clear_console_action)
    
        tools_menu.addSeparator()
    
        restart_action = QAction("Restart", self)
        restart_action.setShortcut("Ctrl+Shift+R")
        restart_action.triggered.connect(self.restart_app)
        tools_menu.addAction(restart_action)
    
        # Help Menu
        help_menu = menubar.addMenu("Help")
    
        user_guide_action = QAction("User Guide", self)
        user_guide_action.setShortcut("F1")
        user_guide_action.triggered.connect(self.show_help)
        help_menu.addAction(user_guide_action)
    
        about_action = QAction("About Neurolyzer", self)
        about_action.triggered.connect(lambda: self.log_action(
            """Neurolyzer v1.0
             Developed by the Cellular Neurophysiology Project Team at Western Sydney University (2025).
             
             Neurolyzer is a cross-platform desktop application for analysing and visualising neurophysiology recordings in ABF format.
             
             Key Features:
             - Extracellular signal analysis (FFT, filtering, plotting)
             - Resonance frequency and ZAP profiling
             - K+ Clearance and Tau calculation
             - Interactive GUI with console, session export, and user settings
             - DuckDB + Parquet integration for structured data management
             
             Powered by: PyQt5, pyABF, NumPy, SciPy, neuroDSP, DuckDB
             """)
        )
        help_menu.addAction(about_action)
        
    def log_action(self, message):
        if hasattr(self, 'TabbedFunction'):
            self.TabbedFunction.AppendResults(message)
        else:
            print(message)

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
        self.loaded_file_label = QLabel("No file loaded")
        self.console_message_label = QLabel("Ready")
        
        # on the bottom right
        self.status_bar.addPermanentWidget(self.loaded_file_label)
        # on the bottom left
        self.status_bar.addWidget(self.console_message_label)
    
    def update_status_bar(self, filename=None, console_message=None):
        if filename is not None:
            self.loaded_file_label.setText(f"Loaded: {filename}")
        if console_message is not None:
            self.console_message_label.setText(console_message)
             
    #UI Functions
    def setup_ui(self):
        # Central widget
        Mainlayout = QWidget()
        self.setCentralWidget(Mainlayout)

        # Ensure central widget fills space
        Mainlayout.setContentsMargins(0, 0, 0, 0)
        Mainlayout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add a dummy layout to Mainlayout
        dummy_layout = QVBoxLayout()
        dummy_layout.setContentsMargins(0, 0, 0, 0)
        dummy_layout.setSpacing(0)
        Mainlayout.setLayout(dummy_layout)
        
        # Sets up Widgets and handlers
        self.TabbedFunction = TabbedWindow(self.update_status_bar)
        self.Directory = DirectoryWidget()
        self.SettingWindow = Settings(parent=self)
        self.HelpBook = HelpBook(parent=self)
        self.PlotArea = IxDPlotArea()
        self.Shortcuts = ShortcutWidget(parent=self)
        # sets up Handlers
        self.ResultHandler = ResultHandler(self.TabbedFunction, self.PlotArea, self.segment, self.heatmap, self.spectrum, self.results, self.zap, self.clearance)    
        self.MessageHandler = MessageHandler(self.TabbedFunction,  self.controller)
        self.ErrorHandler = ErrorHandler(self.TabbedFunction)
        self.TypeSelector = AnalysisSelector(
            controller=self.controller,
            result_handler=self.ResultHandler,
            message_handler=self.MessageHandler,
            error_handler=self.ErrorHandler
        )       
    
        # Create Docks
        self.left_dock = QDockWidget("Tools & Controls", self)
        self.left_dock.setWidget(self.TypeSelector)
        self.left_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.left_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.left_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        #self.left_panel.setMinimumWidth(400)
        
        self.directory_dock = QDockWidget("Directory", self)
        self.directory_dock.setWidget(self.Directory)
        self.directory_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.directory_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.directory_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.shortcut_dock = QDockWidget("Shortcuts", self)
        self.shortcut_dock.setWidget(self.Shortcuts)
        self.shortcut_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.shortcut_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.shortcut_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.plot_dock = QDockWidget("Plot Area", self)
        self.plot_dock.setWidget(self.PlotArea)
        self.plot_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.plot_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.plot_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.plot_dock.setMinimumWidth(350)
    
        self.tabs_dock = QDockWidget("Console / Results", self)
        self.tabs_dock.setWidget(self.TabbedFunction)
        self.tabs_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.tabs_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.tabs_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
       # self.tabs_dock.setMinimumHeight(180)
        
        # Add Docks to Main Window 
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.directory_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.shortcut_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plot_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tabs_dock)
        
        self.splitDockWidget(self.left_dock, self.directory_dock, Qt.Vertical)
        self.splitDockWidget(self.directory_dock, self.shortcut_dock, Qt.Vertical)
        self.splitDockWidget(self.plot_dock, self.tabs_dock, Qt.Vertical)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_dock_sizes()
    
    def adjust_dock_sizes(self):
        # Horizontal split between left and plot
        self.resizeDocks(
            [self.left_dock, self.plot_dock],
            [300, 1980],
            Qt.Horizontal
        )
        
        # Vertical split plot and tabs
        self.resizeDocks(
            [self.plot_dock, self.tabs_dock],
            [5, 2],
            Qt.Vertical
        )
        # Vertical stack on left (if separate docks)
        self.resizeDocks(
            [self.left_dock, self.directory_dock, self.shortcut_dock],
            [2, 1, 1],
            Qt.Vertical
        )

    def show_settings(self):
        if self.SettingWindow:
            self.SettingWindow.show()

    def clear_console(self):
        if hasattr(self, 'TabbedFunction'):
            self.TabbedFunction.ClearMessage()

    # Restores Saved User settings
    def restore_user_settings(self):
        settings = QSettings("WUSoftware", "Neurolyzer")
        if settings.value("appearance/full_screen_start", True, type=bool):
            self.showMaximized()
        else:
            self.showNormal()
        self.directoryPath = settings.value("application/Directory_Path", "", type=str)
        
    # For Importing files
    def browse_file(self):
        # Opens file directory to select ABF file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ABF File", self.directoryPath, "ABF Files (*.abf)")
        # Adds file if selected
        if file_path:
            # saved variable in Self instance
            self.selected_file = file_path
            self.filtered_signal = None
            self.filtered_time = None
            self.filtered_fs = None
            self.TypeSelector.set_selected_file(self.selected_file)
            self.TabbedFunction.AppendMessage(f"[PROGRESS] Loading ABF file: {file_path}")

            # For Debugging Purposes
            # print(f"[DEBUG] file_path type: {type(self.selected_file)} - {self.selected_file}")
            
            #Creates thread to do the task. 
            self.controller.setup_task(self.selected_file, task = "ABFLoading")
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_Plot,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
            
            self.TypeSelector.SelectCurrentTab()
            self.update_status_bar(filename=os.path.basename(file_path), console_message="File loaded and ready.")
        else:
            self.TabbedFunction.AppendMessage("[ERROR] No file selected")
            
    def save_session(self):
        # Ensure there is a file loaded before saving session
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "No ABF file loaded. Cannot save session.")
            return
        
        print("First Print")
        print(f"Segments saved: {self.segment}")
        print(f"Heatmap saved: {self.heatmap}")
        print(f"Spectrum saved: {self.spectrum}")
        print(f"Saving results: {self.results}")
        print(f"Zap saved: {self.zap}")
        print(f"Clearance saved: {self.clearance}")
        
        session_data = {
            "ABF_file_path": self.selected_file,
        }
        
        if ((self.segment and any(self.segment.values())) or (self.heatmap and any(self.heatmap.values())) or (self.spectrum and any(self.spectrum.values()))):
            session_data.update({
                "Results": self.results,
                "Segment": self.segment,
                "Heatmap": self.heatmap,
                "Spectrum": self.spectrum
            })
            
            self.zap = {
                "Frequency": [],
                "Impedance": []
            }
            
            self.clearance = {
                "peak": [],
                "t0": [],
                "v_baseline": [],
                "v_peak": [],
                "delta_v": [],
                "k_baseline": [],
                "k_peak": [],
                "delta_k": [],
                "tau": [],
                "lambda": [],
                "r2": [],
                "amps": [],
                "C": [],
                "decay_time": [],
                "slice_x": [],
                "slice_y": [],
                "fit_x": [],
                "fit_y": [],
                "k90": [],
                "k10": [],
                "full_x": [],
                "full_y": [],
                "t_win": [],
                "y_win": [],
                "n_data": [],
                "slice_t0": [],
                "slice_dt": [],
                "seg_x": [],
                "seg_y": [],
                "fitted": [],
                "v90": [],
                "v10": [],
                "tau_s_str": [],
                "lam_s_str": [],
                "delay_time": [],
                "A_str": []
            }
            
        elif self.zap and (self.zap.get("Frequency") or self.zap.get("Impedance")):
            session_data.update({
                "Results": self.results,
                "Zap_profile": self.zap
            })
            
            self.segment = {
                "Baseline": [],
                "HighK": [],
                "Recovery": []
            }
            
            self.heatmap = {
                "labels": [],
                "f": [],
                "t": [],
                "Sxx": [],
                "freq": [],
                "time": [],
                "power": [],
                "Spec_min": [],
                "Spec_max": []
            }
            
            self.spectrum = {
                "labels": [],
                "freqs": [],
                "mean_psd": [],
                "sem_psd": []
            }
            
            self.clearance = {
                "peak": [],
                "t0": [],
                "v_baseline": [],
                "v_peak": [],
                "delta_v": [],
                "k_baseline": [],
                "k_peak": [],
                "delta_k": [],
                "tau": [],
                "lambda": [],
                "r2": [],
                "amps": [],
                "C": [],
                "decay_time": [],
                "slice_x": [],
                "slice_y": [],
                "fit_x": [],
                "fit_y": [],
                "k90": [],
                "k10": [],
                "full_x": [],
                "full_y": [],
                "t_win": [],
                "y_win": [],
                "n_data": [],
                "slice_t0": [],
                "slice_dt": [],
                "seg_x": [],
                "seg_y": [],
                "fitted": [],
                "v90": [],
                "v10": [],
                "tau_s_str": [],
                "lam_s_str": [],
                "delay_time": [],
                "A_str": []
            }
            
        elif self.clearance and any(self.clearance.values()):
            session_data.update({
                "Results": self.results,
                "Clearance" : self.clearance
            })
            
            self.segment = {
                "Baseline": [],
                "HighK": [],
                "Recovery": []
            }
            
            self.heatmap = {
                "labels": [],
                "f": [],
                "t": [],
                "Sxx": [],
                "freq": [],
                "time": [],
                "power": [],
                "Spec_min": [],
                "Spec_max": []
            }
            
            self.spectrum = {
                "labels": [],
                "freqs": [],
                "mean_psd": [],
                "sem_psd": []
            }
            
            self.zap = {
                "Frequency": [],
                "Impedance": []
            }
        
        print("Second Print")
        print(f"Segments saved: {self.segment}")
        print(f"Heatmap saved: {self.heatmap}")
        print(f"Spectrum saved: {self.spectrum}")
        print(f"Saving results: {self.results}")
        print(f"Zap saved: {self.zap}")
        print(f"Clearance saved: {self.clearance}")

        # Let the user select a directory to save the session
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session As...",
            "session_data.pkl",  # Default filename
            "Pickle Files (*.pkl);;All Files (*)"
        )

        if not save_path:
            return  # User cancelled

        try:
            joblib.dump(session_data, save_path)  # Save session data (ABF file path and analysis results)
            QMessageBox.information(self, "Session Saved", f"Session saved successfully to:\n{save_path}")
            self.TabbedFunction.AppendMessage(f"[PROGRESS] Session saved to {save_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save session:\n{str(e)}")
            self.TabbedFunction.AppendMessage(f"[Error] Unable to save session: {str(e)}")
        
    def load_session(self):
        # Let the user select a session file to load
        session_file, _ = QFileDialog.getOpenFileName(self, "Load Session", self.directoryPath, "Session Files (*.pkl)")
        if not session_file:
            return  # User cancelled

        # Load the session data from the file
        session_data = joblib.load(session_file)

        # Retrieve the ABF file path and results from the session
        abf_file_path = session_data.get("ABF_file_path")
        seg_extract = session_data.get("Segment")
        heatmap_trace = session_data.get("Heatmap")
        power_density = session_data.get("Spectrum")
        loaded_results = session_data.get("Results")
        zap_profile = session_data.get("Zap_profile")
        clearance = session_data.get("Clearance")

        if abf_file_path:
            # Restore the ABF file path
            self.selected_file = abf_file_path
            self.TypeSelector.set_selected_file(self.selected_file)
            
            # Ensure required structures are not None
            self.segment = seg_extract if isinstance(seg_extract, tuple) else {
                "Baseline": [],
                "HighK": [],
                "Recovery": []
            }
            self.heatmap = heatmap_trace if isinstance(heatmap_trace, tuple) else {
                "labels": [],
                "f": [],
                "t": [],
                "Sxx": [],
                "freq": [],
                "time": [],
                "power": [],
                "Spec_min": [],
                "Spec_max": []
            } 
            self.spectrum = power_density if isinstance(power_density, tuple) else {
                "labels": [],
                "freqs": [],
                "mean_psd": [],
                "sem_psd": []
            }
            self.zap = zap_profile if isinstance(zap_profile, tuple) else {
                "Frequency": [],
                "Impedance": []
            }
            self.clearance = clearance if isinstance(clearance, tuple) else {
                "peak": [],
                "t0": [],
                "v_baseline": [],
                "v_peak": [],
                "delta_v": [],
                "k_baseline": [],
                "k_peak": [],
                "delta_k": [],
                "tau": [],
                "lambda": [],
                "r2": [],
                "amps": [],
                "C": [],
                "decay_time": [],
                "slice_x": [],
                "slice_y": [],
                "fit_x": [],
                "fit_y": [],
                "k90": [],
                "k10": [],
                "full_x": [],
                "full_y": [],
                "t_win": [],
                "y_win": [],
                "n_data": [],
                "slice_t0": [],
                "slice_dt": [],
                "seg_x": [],
                "seg_y": [],
                "fitted": [],
                "v90": [],
                "v10": [],
                "tau_s_str": [],
                "lam_s_str": [],
                "delay_time": [],
                "A_str": []
            }
            
            print("\n")
            print("Check this", clearance)
            print("\n")
            
            # Clear and update self.results in-place so ResultHandler still sees the changes
            self.results.clear()
            if isinstance(loaded_results, dict):
                self.results.update(loaded_results)
                print(f"1st Saving results is now: {self.results}")

            # Update the ResultHandler (this is optional if it still points to self.results)
            self.ResultHandler.segment = self.segment
            self.ResultHandler.heatmap = self.heatmap
            self.ResultHandler.spectrum = self.spectrum
            self.ResultHandler.results = self.results
            self.ResultHandler.zap = self.zap
            self.ResultHandler.clearance = self.clearance
                
            # Refresh GUI to reflect restored results
            self.update_results_display()
            
            # If you need to re-run the calculations or replot, you can trigger those actions here:
            # For example:
            self.controller.setup_task(self.selected_file, task="ABFLoading")
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_Plot,  # Assuming you want to plot results after loading
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
            
            if seg_extract and any(seg_extract[key] for key in seg_extract):
                self.controller.setup_task(self.selected_file, task = "SegmentEcellularLoadingSession", baselinedata = seg_extract["Baseline"], highkdata = seg_extract["HighK"], recoverydata = seg_extract["Recovery"])
                self.controller.connect_signals({
                    "error": self.ErrorHandler.handle_error,
                    "finished": self.MessageHandler.cleanup_after_thread
                })
                
            if heatmap_trace and any(heatmap_trace[key] for key in heatmap_trace):
                self.controller.setup_task(self.selected_file, task = "HeatmapPlotLoadingSession", heatmapdata = heatmap_trace)
                self.controller.connect_signals({
                    "error": self.ErrorHandler.handle_error,
                    "finished": self.MessageHandler.cleanup_after_thread
                })
                
            if power_density and any(power_density[key] for key in power_density):
                self.controller.setup_task(self.selected_file, task = "SpectrumPlotLoadingSession", spectrumdata = power_density)
                self.controller.connect_signals({
                    "error": self.ErrorHandler.handle_error,
                    "finished": self.MessageHandler.cleanup_after_thread
                })
            
            if zap_profile and (zap_profile.get("Frequency") or zap_profile.get("Impedance")):
                self.controller.setup_task(self.selected_file, task = "GenImpedancePlotLoadingSession")
                self.controller.connect_signals({
                    "error": self.ErrorHandler.handle_error,
                    "finished": self.MessageHandler.cleanup_after_thread
                })
                
            if clearance and any(clearance[key] for key in clearance):
                self.controller.setup_task(self.selected_file, task = "ClearancePlotLoadingSession", clearancedata = clearance)
                self.controller.connect_signals({
                    "error": self.ErrorHandler.handle_error,
                    "finished": self.MessageHandler.cleanup_after_thread
                })
                
            self.controller.setup_task(self.selected_file, task = "Set Current Tab")
            self.controller.connect_signals({
                "result": self.TypeSelector.HandleTabData,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
            
            self.results = {
                "Analysis Type": [],
                "Value": []
            }
            self.ResultHandler.results = self.results
            print(f"2nd Saving results is now: {self.ResultHandler.results}")
        else:
            QMessageBox.warning(self, "Invalid Session", "The session file is missing required data.")
            self.TabbedFunction.AppendMessage("[ERROR] Invalid session file format or missing data.")

    def update_results_display(self):
        """Refresh the GUI to show the currently loaded results."""
        analysis_list = self.results.get("Analysis Type", [])
        value_list = self.results.get("Value", [])

        if not analysis_list or not value_list:
            self.TabbedFunction.AppendMessage("No analysis results to display.")
            return

        for analysis, value in zip(analysis_list, value_list):
            self.TabbedFunction.AppendResults(f"{analysis}: {value}")
            
    def restart_app(self):
        QApplication.quit()
        subprocess.Popen([sys.executable,  sys.argv[0], *sys.argv[1:]])
        # os.execl(sys.executable, sys.executable, *sys.argv)
        sys.exit()
        
    def show_help(self):
        if self.HelpBook:
            self.HelpBook.show()
        
    def quit_app(self):
        QApplication.quit()
    
    def export_abf_to_csv(self):
        if not self.selected_file:
            self.TabbedFunction.AppendMessage("[ERROR] No file selected!")
            return

        abf = pyabf.ABF(self.selected_file)
        abf.setSweep(0)
        time_data = abf.sweepX
        voltage_data = abf.sweepY

        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", self.directoryPath, "CSV Files (*.csv)")
        if save_path:
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Voltage (mV)"])
                for t, v in zip(time_data, voltage_data):
                    writer.writerow([t, v])
            self.TabbedFunction.AppendMessage(f"[PROGRESS] Exported ABF data to CSV: {save_path}")