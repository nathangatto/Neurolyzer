from PyQt5.QtCore import QObject, pyqtSignal
from Backend.Services.ABFfileLoader import FileLoader
from Backend.Services.ABFfileLoader import FileSession
from Backend.Services.Ecellular import Extracellular
from Backend.Services.ResonanceFreq import ResonanceFreq
from Backend.Services.clearance import ClearanceSession

class NeuroThreadWorker(QObject):
    #Defines all signals possible
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    validation = pyqtSignal(dict)
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.file_path = args[0]
        self.task = kwargs.get("task")
        self.baselinedata = kwargs.get("baselinedata")
        self.highkdata = kwargs.get("highkdata")
        self.recoverydata = kwargs.get("recoverydata")
        self.heatmap = kwargs.get("heatmapdata")
        self.spectrum = kwargs.get("spectrumdata")
        self.clearance = kwargs.get("clearancedata")
        # default 50 Hz if the caller does not specify mains
        self.mains = kwargs.get("mains", 50)
        self.fit_method = kwargs.get("fit_method", "lm")
        self.sweep = kwargs.get("sweep", 0)
        from Backend.Services.clearance import DEFAULT_SLOPE, RESTING_K_MM
        self.slope = kwargs.get("slope", DEFAULT_SLOPE)
        self.resting_k = kwargs.get("resting_k", RESTING_K_MM)
        self.pre_calibration = kwargs.get("pre_calibration", None)
        self.post_calibration = kwargs.get("post_calibration", None)
        self.validate = kwargs.get("validate", True)
        self.FreqRange = kwargs.get("Range", 0)
        self.n_exp = kwargs.get("n_exp", 2)
        self.BandPassSelected = kwargs.get("bandpassfilter")
        self.HighPassSelected = kwargs.get("highpassfilter")
        self.LowPassSelected = kwargs.get("lowpassfilter")
        
    def run(self):
        try:
            # For Debugging uses only!
            print(f"[DEBUG] file_path type: {type(self.file_path)} - {self.file_path}")
            print(f"[DEBUG] Task type: {self.task}")
            
            if self.task == "ABFLoading":
                Extracellular.clear()
                session = FileLoader(self.file_path)
                x , y , current = session.PlotRawABF()
                result = (x , y, current)
            elif self.task == "Set Current Tab":
                session = FileLoader(self.file_path)
                session.GetFileInfo()
                result = session.DetemineAnalysisType()
            elif self.task == "SegmentEcellular":
                segmenter = Extracellular(self.file_path)
                print(self.BandPassSelected, self.HighPassSelected, self.LowPassSelected)
                segments = segmenter.segment_raw_signal(
                    baseline_data=self.baselinedata, 
                    highk_data=self.highkdata, 
                    recovery_data=self.recoverydata,
                    bandpassfilter= self.BandPassSelected,
                    highpassfilter= self.HighPassSelected,
                    lowpassfilter= self.LowPassSelected
                    )
                result = segments
                print(result)
            elif self.task == "SegmentEcellularLoadingSession":
                result = ("SegmentEcellular", 
                {
                    "baseline": self.baselinedata,
                    "high_k": self.highkdata,
                    "recovery": self.recoverydata
                })
            elif self.task == "HeatmapPlot":
                Heatmap = Extracellular(self.file_path)
                if Extracellular.get_stored_segments():
                    segments = Extracellular.get_stored_segments()
                    spectograms = Heatmap.Extract_SpectogramData(segments)
                    global_min, global_max = Heatmap.Calc_globalMinMax(spectograms)
                    print(global_min, global_max)
                    result = Heatmap.display_heatmap(spectograms=spectograms, global_min=global_min, global_max=global_max)
                    print(result)
                else:
                    raise ValueError("No segmented data available. Run segmenting first.")
            elif self.task == "HeatmapPlotLoadingSession":
                result = ("heatmap_trace", 
                {
                    label: (f, t, Sxx, Spec_min, Spec_max)
                    for label, (f, t, Sxx, Spec_min, Spec_max) in zip(self.heatmap["labels"], zip(self.heatmap["f"], self.heatmap["t"], self.heatmap["Sxx"], self.heatmap["Spec_min"], self.heatmap["Spec_max"]))
                })
            elif self.task == "SpectrumPlot":
                Spectrum = Extracellular(self.file_path)
                if Extracellular.is_available():
                    result = Spectrum.display_spectrum(Extracellular.filtered_signal, Extracellular.filtered_time, Extracellular.filtered_fs)
                else:
                    result = Spectrum.display_spectrum()
            elif self.task == "SpectrumPlotLoadingSession":
                result = ("spectral_density", 
                {
                    label: {
                        "freqs": f,
                        "mean": mean,
                        "sem": sem
                    }
                    for label, f, mean, sem in zip(
                        self.spectrum["labels"], 
                        self.spectrum["freqs"], 
                        self.spectrum["mean_psd"], 
                        self.spectrum["sem_psd"]
                    )
                })
            elif self.task == "GenImpedancePlot":
                GenZap = ResonanceFreq(self.file_path, self.FreqRange)
                GenZap.ExtractData()
                GenZap.CalcFFT()
                freq, impedance = GenZap.CalcImpedance()
                result = (freq, impedance)
            elif self.task == "GenImpedancePlotLoadingSession":
                GenZap = ResonanceFreq(self.file_path, self.FreqRange)
                GenZap.ExtractData()
                GenZap.CalcFFT()
                freq, impedance = GenZap.CalcImpedance()
                result = ("zap_profile", freq, impedance)
            elif self.task == "CalcResonanceFreq":
                CalcFreq = ResonanceFreq(self.file_path, self.FreqRange)
                CalcFreq.ExtractData()
                CalcFreq.CalcFFT()
                CalcFreq.CalcImpedance()
                ResonanceFrequency = CalcFreq.CalcResonanceFreq()
                result = ResonanceFrequency
            elif self.task == "CalcHWA":
                CalcFreq = ResonanceFreq(self.file_path, self.FreqRange)
                CalcFreq.ExtractData()
                CalcFreq.CalcFFT()
                CalcFreq.CalcImpedance()
                f_left, f_right, z_left, z_right, HWA = CalcFreq.CalcHWA()
                result = (f_left, f_right, z_left, z_right, HWA)
            elif self.task == "SegmentEpisodes":
                sess = ClearanceSession(
                    self.file_path,
                    sweep=self.sweep,
                    fit_method=self.fit_method,
                    slope=self.slope,
                    resting_k=self.resting_k,
                    n_exp=self.n_exp
                )

                sess.apply_notch(self.mains)
                sess.detect_peaks()
                sess._fit_method = self.fit_method
                results = sess.clearance_test() 

                validation_results = None
                if self.validate:
                    validation_results = sess.validate_experiment(  
                        pre_calibration=self.pre_calibration, 
                        post_calibration=self.post_calibration
                    )
                    
                    self.validation.emit(validation_results)
                    self._log_validation(validation_results)

                self.result.emit(("clearance", results, sess, validation_results))
            if self.task == "ClearancePlotLoadingSession":
                # Check if we have pre-loaded clearance data
                if hasattr(self, 'clearance') and self.clearance is not None:
                    # Teammate's code - loading from saved data
                    def dict_of_lists_to_list_of_dicts(clearance_data):
                        n = len(clearance_data["peak"])  # assumes all lists have equal length
                        list_of_dicts = []
                        for i in range(n):
                            segment_dict = {key: clearance_data[key][i] for key in clearance_data}
                            list_of_dicts.append(segment_dict)
                        return list_of_dicts
                    
                    result = {
                        "peak": self.clearance["peak"],
                        "t0": self.clearance["t0"],
                        "v_baseline": self.clearance["v_baseline"],
                        "v_peak": self.clearance["v_peak"],
                        "delta_v": self.clearance["delta_v"],
                        "k_baseline": self.clearance["k_baseline"],
                        "k_peak": self.clearance["k_peak"],
                        "delta_k": self.clearance["delta_k"],
                        "tau": self.clearance["tau"],
                        "lambda": self.clearance["lambda"],
                        "r2": self.clearance["r2"],
                        "amps": self.clearance["amps"],
                        "C": self.clearance["C"],
                        "decay_time": self.clearance["decay_time"],
                        "slice_x": self.clearance["slice_x"],
                        "slice_y": self.clearance["slice_y"],
                        "fit_x": self.clearance["fit_x"],
                        "fit_y": self.clearance["fit_y"],
                        "k90": self.clearance["k90"],
                        "k10": self.clearance["k10"],
                        "full_x": self.clearance["full_x"],
                        "full_y": self.clearance["full_y"],
                        "t_win": self.clearance["t_win"],
                        "y_win": self.clearance["y_win"],
                        "n_data": self.clearance["n_data"],
                        "slice_t0": self.clearance["slice_t0"],
                        "slice_dt": self.clearance["slice_dt"],
                        "seg_x": self.clearance["seg_x"],
                        "seg_y": self.clearance["seg_y"],
                        "fitted": self.clearance["fitted"],
                        "v90": self.clearance["v90"],
                        "v10": self.clearance["v10"],
                        "tau_s_str": self.clearance["tau_s_str"],
                        "lam_s_str": self.clearance["lam_s_str"],
                        "delay_time": self.clearance["delay_time"],
                        "A_str": self.clearance["A_str"],
                    }
                    clearance_list = dict_of_lists_to_list_of_dicts(result)
                    
                    # Create a dummy session for compatibility
                    sess = ClearanceSession(
                        self.file_path,
                        sweep=self.sweep,
                        fit_method=self.fit_method,
                        slope=self.slope,
                        resting_k=self.resting_k,
                        n_exp=self.n_exp
                    )
                    sess.apply_notch(self.mains)
                    sess.results = clearance_list
                    
                    # Emit with session for your code compatibility
                    self.result.emit(("clearance1", clearance_list, sess))
                    
                else:
                    # Your original code - run fresh analysis
                    sess = ClearanceSession(
                        self.file_path,
                        sweep=self.sweep,
                        fit_method=self.fit_method,
                        slope=self.slope,
                        resting_k=self.resting_k,
                        n_exp=self.n_exp
                    )
                    sess.apply_notch(self.mains)
                    sess.detect_peaks()
                    sess._fit_method = self.fit_method
                    results = sess.clearance_test()
                    sess.results = results
                    self.result.emit(("clearance1", results, sess))
            elif self.task in ("Filter50Hz", "Filter60Hz"):
                mains = 50 if self.task == "Filter50Hz" else 60
                sess = ClearanceSession(
                            self.file_path,
                            sweep     = self.sweep,
                            fit_method= self.fit_method,
                            slope     = self.slope,
                            resting_k = self.resting_k
                        )
                sess.apply_notch(freq=mains)
                result = ("filtered", mains, sess.x, sess.filtered)
            elif self.task == "ShowFullSignal":
                sess = ClearanceSession(
                            self.file_path,
                            sweep     = self.sweep,
                            fit_method= self.fit_method,
                            slope     = self.slope,
                            resting_k = self.resting_k,
                            fit_domain="voltage",
                            n_exp=self.n_exp,
                        )
                sess.apply_notch(self.mains)
                sess.detect_peaks()
                x, conc, peaks, t0 = sess.full_signal()
                result = ("full_signal", x, conc, peaks)
                   
            self.result.emit(result)  
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _log_validation(self, validation_results):
        """Helper method to log validation results"""
        if validation_results['valid']:
            print("✅ Experiment passed all validation checks")
        else:
            print("❌ Experiment failed validation:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("⚠️ Warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")