from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QLabel, QFrame, QPushButton, QFileDialog, QTabWidget, QDoubleSpinBox, QSpinBox
import pyabf
from Backend.Services.clearance import DEFAULT_SLOPE, RESTING_K_MM
import pyqtgraph as pg
#Handler Imports
from Frontend.Handlers.ErrorHandler import ErrorHandler
from Frontend.Handlers.MessageHandler import MessageHandler
from Frontend.Handlers.ResultsHandler import ResultHandler

class AnalysisSelector(QWidget):  # Inheriting QWidget instead of QMainWindow
    def __init__(self, 
                 controller,
                 *,
                 result_handler: ResultHandler,
                 message_handler: MessageHandler,
                 error_handler: ErrorHandler):

        super().__init__()
        self.analysis_buttons = []
        self.E_AnalysisBtn = []
        self.RF_AnalysisBtn = []
        self.C_AnalysisBtn = []
        self.controller = controller
        self.ResultHandler = result_handler
        self.MessageHandler = message_handler
        self.ErrorHandler = error_handler
        self.selectedfile = None
        self.FreqValue = None
        self.mains_freq = 50  # Default to 50Hz
        self.current_sweep = 0
        self.BandPassSelected = False
        self.HighPassSelected = False
        self.LowPassSelected = False

        # Define main horizontal layout for the whole widget
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)   # Directly set the layout for this widget
        
        self.tabs = QTabWidget()
        self.layout().addWidget(self.tabs)
        
        self.tabs.addTab(self.create_Extracellular_tab(), "Extracellular")
        self.tabs.addTab(self.create_Resonance_Frequency_tab(), "Resonance Frequency")
        self.tabs.addTab(self.create_K_Clearance_tab(), "K+ Clearance")
        
        tools_layout = self.UniversalTools()
        
        main_layout.addLayout(tools_layout)

        self.setMinimumSize(357, 315)
    
    def _on_sweep_changed(self, idx: int):
        self.current_sweep = idx
        print(f"[DEBUG] user selected sweep {idx}")    
    
    # Widget Creation
    def UniversalTools(self):
        ToolsLayout = QVBoxLayout()
        FirstRow = QHBoxLayout()
        SecondRow = QHBoxLayout()
        ToolsLayout.setSpacing(0) 
        ToolsLayout.setContentsMargins(0, 0, 0, 0)
        
        self.BypassButtonSelector = QPushButton("Reveal Greyed-out Buttons")
        self.BypassButtonSelector.clicked.connect(self.RemoveGreyOut)
        
        self.BandPassFilter = QPushButton("BandPass")
        self.BandPassFilter.setCheckable(True)
        self.BandPassFilter.clicked.connect(self.Apply_Filter)
        
        self.HighPassFilter = QPushButton("HighPass")
        self.HighPassFilter.setCheckable(True)
        self.HighPassFilter.clicked.connect(self.Apply_Filter)
        
        self.LowPassFilter = QPushButton("LowPass")
        self.LowPassFilter.setCheckable(True)
        self.LowPassFilter.clicked.connect(self.Apply_Filter)
        
        FirstRow.addWidget(self.BypassButtonSelector)    
        SecondRow.addWidget(self.BandPassFilter)
        SecondRow.addWidget(self.HighPassFilter)
        SecondRow.addWidget(self.LowPassFilter)
        ToolsLayout.addLayout(FirstRow)
        ToolsLayout.addLayout(SecondRow)
        return ToolsLayout
    
    def RemoveGreyOut(self):
        if self.selectedfile:
            for btn in self.RF_AnalysisBtn:
                btn.setEnabled(True)
            for btn in self.E_AnalysisBtn:
                btn.setEnabled(True)
            for btn in self.C_AnalysisBtn:
                btn.setEnabled(True)
        else:
            self.ErrorHandler.handle_error("Select File First")
            
    def Apply_Filter(self):
        self.BandPassSelected = self.BandPassFilter.isChecked()
        self.HighPassSelected =self.HighPassFilter.isChecked()
        self.LowPassSelected = self.LowPassFilter.isChecked()
        
        print(self.BandPassSelected, self.HighPassSelected, self.LowPassSelected)

    def create_Extracellular_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # layout.setSpacing(1) 
        # layout.setContentsMargins(1, 1, 1, 1) 
        
        SegmentLayout = QHBoxLayout()
            
        E_actions = [
            ("Segment Data", self.EC_Segment),
            ("Unsegment Data", self.Unsegment),
            ("Extract Data", self.EC_Extract),
            ("Unextract Data", self.Unextract_data),
            ("Plot Power Spectrum", self.EC_heatmap),
            ("Plot Power Density", self.EC_spectrum)
        ]
        
        for i, (name, callback) in enumerate(E_actions):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=callback: self.run_analysis(n))
            self.E_AnalysisBtn.append(btn)
            # Adds the first two buttons to the horizontal layout while the rest go to the vertical one
            if i < 4:
                SegmentLayout.addWidget(btn)
            else:
                layout.addWidget(btn)
                
        self.E_AnalysisBtn[1].hide()
        self.E_AnalysisBtn[3].hide()
        # Ensures layouts are both present
        layout.insertLayout(0, SegmentLayout)
            
        return tab

    def create_Resonance_Frequency_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # layout.setSpacing(1) 
        # layout.setContentsMargins(1, 1, 1, 1)
        
        FreqRangeBox = QHBoxLayout()
        FreqRangeBox.addWidget(QLabel("Frequency Range for Zap"))
        self.FreqRange = QDoubleSpinBox()
        self.FreqRange.setRange(0, 100)
        self.FreqRange.setValue(20)
        self.FreqValue = self.FreqRange.value()
        self.FreqRange.valueChanged.connect(self.update_FreqValue)
        FreqRangeBox.addWidget(self.FreqRange)
        layout.addLayout(FreqRangeBox)
        
        RF_actions = [
            ("Generate ZAP Profile", self.GenImpedancePlot),
            ("Calculate Resonance Frequency", self.CalcResonanceFreq),
            ("Calculate HWA", self.CalcHWA)
        ]
        
        for name, callback in RF_actions:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=callback: self.run_analysis(n))
            layout.addWidget(btn)
            self.RF_AnalysisBtn.append(btn)
            
        return tab

    def create_K_Clearance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # layout.setSpacing(1) 
        # layout.setContentsMargins(1, 1, 1, 1)

        picker_bar       = QHBoxLayout()
        self.sweep_label = QLabel("Sweep:")
        self.sweep_combo = QComboBox()
        self.sweep_combo.currentIndexChanged[int].connect(self._on_sweep_changed)
        picker_bar.addWidget(self.sweep_label)
        picker_bar.addWidget(self.sweep_combo)
        layout.addLayout(picker_bar)


        n_exp_bar = QHBoxLayout()
        n_exp_bar.addWidget(QLabel("Exponential Components:"))
        self.n_exp_spin = QSpinBox()
        self.n_exp_spin.setMinimum(1)
        self.n_exp_spin.setMaximum(2)
        self.n_exp_spin.setValue(2)
        self.n_exp_spin.setToolTip("Number of exponential components")
        n_exp_bar.addWidget(self.n_exp_spin)
        layout.addLayout(n_exp_bar)

        slope_bar = QHBoxLayout()
        slope_bar.addWidget(QLabel("Slope (mV/decade):"))
        self.slope_input = QDoubleSpinBox()
        self.slope_input.setRange(40.0, 80.0)
        self.slope_input.setValue(DEFAULT_SLOPE)
        slope_bar.addWidget(self.slope_input)
        layout.addLayout(slope_bar)

        k0_bar = QHBoxLayout()
        k0_bar.addWidget(QLabel("Resting [K⁺] (mM):"))
        self.k0_input = QDoubleSpinBox()
        self.k0_input.setRange(1.0, 10.0)
        self.k0_input.setValue(RESTING_K_MM)
        k0_bar.addWidget(self.k0_input)
        layout.addLayout(k0_bar)
        
        C_actions = [
            ("Analyze K⁺ Clearance", self.segment_episodes_cheb),
            ("Show Peaks", self.show_full_signal),
            ("Filter 50Hz", self.filter_50hz),
            ("Filter 60Hz", self.filter_60hz),
            ("Compare Traces",       self.compare_traces),
            # ("Export Results", self.export_clearance_results)
        ]
        
        for name, callback in C_actions:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=callback: self.run_analysis(n))
            layout.addWidget(btn)
            self.C_AnalysisBtn.append(btn)
            
        return tab
    
    # Runs lambda function for the analysis
    def run_analysis(self, func):
        # Placeholder for running actual analysis
        func()
    # Sets the Loaded_File
    def set_selected_file(self, file_path: str):
        """
        Called by your file‑open dialog.
        If the user picks a new .abf, reset the plot area and results pane.
        """
        if not file_path:
            print("No file selected")
            return

        # Only reset if the path actually changed
        if getattr(self, "selectedfile", None) != file_path:
            # 1) wipe all segment tabs + raw plot
            self.ResultHandler.reset_display()

            # 2) clear Results and Console panes
            self.ResultHandler.logger.clear_all()

            for name in ("result", "error", "finished"):
                self.controller.clear_signal(name)

        # finally store the new active file
        self.selectedfile = file_path
                # ------------------ sweep picker ----------------
        abf = pyabf.ABF(file_path)
        self.sweep_combo.blockSignals(True)
        self.sweep_combo.clear()

        if abf.sweepCount > 1:
            for i in range(abf.sweepCount):
                self.sweep_combo.addItem(f"Sweep {i}", i)
            self.sweep_combo.setEnabled(True)
            self.sweep_label.setEnabled(True)
        else:                                         # single-sweep file
            self.sweep_combo.addItem("0")
            self.sweep_combo.setEnabled(False)
            self.sweep_label.setEnabled(False)

        self.current_sweep = 0
        self.sweep_combo.blockSignals(False)
        
    # Ensures the current Tab is selected
    def SelectCurrentTab(self):        
        self.controller.setup_task(self.selectedfile, task = "Set Current Tab")
        self.controller.connect_signals({
            "result": self.HandleTabData,
            "error": self.ErrorHandler.handle_error,
            "finished": self.MessageHandler.cleanup_after_thread
        })
    
    # Only Results handle left out of the Result Handler
    def HandleTabData(self, Tabresult):
        self.ResultHandler.printAnalysisType(Tabresult)
        # Checks to see what type of analysis needs to be done and ensures the correct buttons are usable and other are greyed out
        if Tabresult == "K+ Clearance":
            self.tabs.setCurrentIndex(2)
            for btn in self.RF_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.E_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.C_AnalysisBtn:
                btn.setEnabled(True)
        elif Tabresult == "Resonance Frequency":
            self.tabs.setCurrentIndex(1)
            for btn in self.C_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.E_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.RF_AnalysisBtn:
                btn.setEnabled(True)
        elif Tabresult  == "Extracellular":
            self.tabs.setCurrentIndex(0)
            for btn in self.C_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.RF_AnalysisBtn:
                btn.setEnabled(False)
            for btn in self.E_AnalysisBtn:
                btn.setEnabled(True)
        else:
            self.ResultHandler.printAnalysisType(Tabresult=None)
            for btn in self.RF_AnalysisBtn:
                btn.setEnabled(True)
            for btn in self.E_AnalysisBtn:
                btn.setEnabled(True)
            for btn in self.C_AnalysisBtn:
                btn.setEnabled(True)
    
    # reset buttons on opening if any issues
    def clear_sidebar_buttons(self):
        # Remove old buttons from the layout
        for btn in self.analysis_buttons:
            self.sidebar_layout.removeWidget(btn)
            btn.deleteLater()
        self.analysis_buttons = []

    # Dummy button function -> tests if button works
    def print_button(self):
        print("Button clicked!")
    
    # Extracellular Functions - All Below
    
    # Extracellular Segment Functionality
    def EC_Segment(self):
        regionSelected = self.ResultHandler.handle_RegionPlotting()
        
        if self.E_AnalysisBtn[0].isVisible() and regionSelected:
            self.E_AnalysisBtn[0].hide()
            self.E_AnalysisBtn[1].show()
        elif regionSelected == False:
            self.E_AnalysisBtn[1].hide()
            self.E_AnalysisBtn[0].show()
    
    # Extracellular Unsegment Functionality
    def Unsegment(self):
        regionUnSelected = self.ResultHandler.handle_RemoveRegion()
        
        if self.E_AnalysisBtn[1].isVisible() and regionUnSelected:
            self.E_AnalysisBtn[1].hide()
            self.E_AnalysisBtn[0].show()
        else:
            self.E_AnalysisBtn[0].hide()
            self.E_AnalysisBtn[1].show()
    
    # Extracellular Extract Button Functionality
    def EC_Extract(self):
        baselineData, highkData, recoveryData = self.ResultHandler.handle_dataFetch()
        # print(baselineData, highkData, recoveryData)
        
        if baselineData is None or highkData is None or recoveryData is None:
            self.ErrorHandler.handle_error("Segment the data first")
            return None
        else:
            if self.E_AnalysisBtn[2].isVisible():
                self.E_AnalysisBtn[2].hide()
                self.E_AnalysisBtn[3].show()
            else:
                self.E_AnalysisBtn[3].hide()
                self.E_AnalysisBtn[2].show()
            
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")

            self.controller.setup_task(self.selectedfile, 
                task = "SegmentEcellular", 
                baselinedata = baselineData, 
                highkdata = highkData, 
                recoverydata = recoveryData, 
                bandpassfilter= self.BandPassSelected,
                highpassfilter= self.HighPassSelected,
                lowpassfilter= self.LowPassSelected
            )
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_ECsegement,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
            
    # Extracellular Unextract Data Functionality
    def Unextract_data(self):
        self.ResultHandler.handle_DataRemoval()
        
        if self.E_AnalysisBtn[2].isVisible():
            self.E_AnalysisBtn[2].hide()
            self.E_AnalysisBtn[3].show()
        else:
            self.E_AnalysisBtn[3].hide()
            self.E_AnalysisBtn[2].show()
            self.E_AnalysisBtn[1].hide()
            self.E_AnalysisBtn[0].show()
    
    # Extracellular Heatmap Functionality
    def EC_heatmap(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")
            
            self.controller.setup_task(self.selectedfile, task = "HeatmapPlot")
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_HeatmapPlot,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ResultHandler.handle_HeatmapPlot(result=None)
    
    # Extracellular Spectral Density Functionality 
    def EC_spectrum(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")
            
            self.controller.setup_task(self.selectedfile, task = "SpectrumPlot")
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_SpectrumPlot,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ResultHandler.handle_SpectrumPlot(result=None)
        
    # Resonance Frequancy Functions - All below
    
    # Updates the Frequency range value 
    def update_FreqValue(self, val):
        self.FreqValue = float(val)

    # Sets up the task: Plots the Impedance Plot
    def GenImpedancePlot(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")

            self.controller.setup_task(self.selectedfile, task = "GenImpedancePlot" , Range= self.FreqValue)
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_ImpedancePlot,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ErrorHandler.handle_error("No ABF file Loaded, Please Open ABF File")
    
    # Sets up the task: Calulcates and prints the Resonance Frequency
    def CalcResonanceFreq(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")

            self.controller.setup_task(self.selectedfile, task = "CalcResonanceFreq", Range= self.FreqValue)
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_CalcResonanceFreq,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ErrorHandler.handle_error("No ABF file Loaded, Please Open ABF File")
        
    # Sets up the task: Calulcates and prints the Half-Width at Attenaution
    def CalcHWA(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")
            
            self.controller.setup_task(self.selectedfile, task = "CalcHWA", Range= self.FreqValue)
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_CalcHWA,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ErrorHandler.handle_error("No ABF file Loaded, Please Open ABF File")
    
    # K+ clearance functions - All below
    def segment_episodes(self):
        if self.selectedfile:
            self.controller.clear_signal("result")
            self.controller.clear_signal("error")
            self.controller.clear_signal("finished")
            
            self.controller.setup_task(self.selectedfile, task = "SegmentEpisodes", mains=self.mains_freq, sweep=self.current_sweep, fit_method="cheb", slope=self.slope_input.value(), resting_k=self.k0_input.value(), n_exp=self.n_exp_spin.value())
            self.controller.connect_signals({
                "result": self.ResultHandler.handle_Clearance_results,
                "error": self.ErrorHandler.handle_error,
                "finished": self.MessageHandler.cleanup_after_thread
            })
        else:
            self.ErrorHandler.handle_error("No ABF file Loaded, Please Open ABF File")
    
    def show_full_signal(self):
        """
        Launch the worker that loads the sweep, runs the notch filter +
        peak‑finder, and emits ('full_signal', x, k_conc, peaks).
        The ResultHandler already knows how to plot that tuple.
        """
        if not getattr(self, "selectedfile", None):
            self.ErrorHandler.handle_error("No ABF file loaded,  Please Open ABF File")
            return
        
        self.controller.clear_signal("result")
        self.controller.clear_signal("error")
        self.controller.clear_signal("finished")
        
        self.controller.setup_task(
            self.selectedfile, 
            task="ShowFullSignal", 
            mains=self.mains_freq, 
            sweep=self.current_sweep, 
            fit_method="cheb", 
            slope=self.slope_input.value(), 
            resting_k=self.k0_input.value(),
            n_exp=self.n_exp_spin.value() )
        self.controller.connect_signals({
            "result":   self.ResultHandler.handle_Plot,
            "error":    self.ErrorHandler.handle_error,
            "finished": self.MessageHandler.cleanup_after_thread,
        })

    def _start_filter_task(self, task_name):
        if not getattr(self, "selectedfile", None):
            self.ErrorHandler.handle_error("No ABF file loaded,  Please Open ABF File")
            return
        
        self.mains_freq = 50 if task_name == "Filter50Hz" else 60
        
        self.controller.clear_signal("result")
        self.controller.clear_signal("error")
        self.controller.clear_signal("finished")
        
        self.controller.setup_task(self.selectedfile, task=task_name, mains=self.mains_freq)
        self.controller.connect_signals({
            "result":   self.ResultHandler.handle_Plot,
            "error":    self.ErrorHandler.handle_error,
            "finished": self.MessageHandler.cleanup_after_thread,
        })
        
    def filter_50hz(self):
        self.mains_freq = 50
        self._start_filter_task("Filter50Hz")
    
    def filter_60hz(self):
        self.mains_freq = 60
        self._start_filter_task("Filter60Hz")

    def segment_episodes_cheb(self):
        """
        Same worker as SegmentEpisodes but with fit_method='cheb'.
        The controller just needs to pass that kwarg through.
        """
        
        self.controller.clear_signal("result")
        self.controller.clear_signal("error")
        self.controller.clear_signal("finished")
        
        self.controller.setup_task(
            self.selectedfile,
            task="SegmentEpisodes",
            mains=self.mains_freq,
            fit_method="cheb",
            slope=self.slope_input.value(),
            resting_k=self.k0_input.value(),
            sweep=self.current_sweep,
            n_exp=self.n_exp_spin.value() 
        )
        self.controller.connect_signals({
            "result":   self.ResultHandler.handle_Clearance_results,
            "error":    self.ErrorHandler.handle_error,
            "finished": self.MessageHandler.cleanup_after_thread
        })

    def export_clearance_results(self):
        # ask where to put the files
        dir_path = QFileDialog.getExistingDirectory(
            self, "Choose export folder", ".", QFileDialog.ShowDirsOnly
        )
        if not dir_path:
            return  # user cancelled

        # delegate to ResultHandler
        self.ResultHandler.export_clearance(dir_path)

    # Compare trace Functionality
    def compare_traces(self):
        """
        Ask for a second .abf file, analyse it exactly like the primary one,
        and overlay its sweep with proper baseline alignment. 
        """
        from Backend.Services.clearance import ClearanceSession
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import os

        fname, _ = QFileDialog.getOpenFileName(
            self, "Choose ABF file to compare", ".", "ABF files (*.abf)")
        if not fname:
            return

        try:
            # run the SAME pipeline as the primary trace
            sess2 = ClearanceSession(
                fname,
                sweep=self.current_sweep,
                fit_method="cheb",
                slope=self.slope_input.value(),
                resting_k=self.k0_input.value(),
                n_exp=self.n_exp_spin.value()
            )
            sess2.apply_notch(self.mains_freq)
            sess2.detect_peaks()
            
            # Get the primary session
            sess1 = self.ResultHandler.PlotArea.session
            
            # Get full signals
            x1, y1, peaks1, t01 = sess1.full_signal()
            x2, y2, peaks2, t02 = sess2.full_signal()
            
            # Run clearance test
            second_results = sess2.clearance_test()
            
        except Exception as e:
            QMessageBox.critical(self, "Error loading comparison trace", str(e))
            return

        # Clear and redraw with overlays
        self.ResultHandler.PlotArea.channel0_Plot.clear()
        
        # For the raw plot, use the zeroed traces
        y1_zeroed = sess1.conc_zeroed
        y2_zeroed = sess2.conc_zeroed
        
        # Plot both traces with baseline at 0
        self.ResultHandler.PlotArea.overlay_trace(x1, y1_zeroed, peaks1, t0=t01, pen=pg.mkPen('k'))
        self.ResultHandler.PlotArea.overlay_trace(x2, y2_zeroed, peaks2, t0=t02)
        
        # Pass the session object for proper baseline correction in segments
        self.ResultHandler.PlotArea.overlay_segments(second_results, sess2)