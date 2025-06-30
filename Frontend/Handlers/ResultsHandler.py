import traceback
from Backend.Services.exporter import results_to_csv, export_clearance_plots
from PyQt5.QtWidgets import QMessageBox
from collections import defaultdict

class ResultHandler:
    def __init__(self, logger_widget, Plot_Widget, seg = None, heatmaptrace = None, density = None, results = None, zapresult = None, clearesult = None, export = None):
        self.logger = logger_widget
        self.PlotArea = Plot_Widget
        self.last_clearance = None
        self.segment = seg
        self.heatmap = heatmaptrace
        self.spectrum = density
        self.results = results
        self.zap = zapresult
        self.clearance = defaultdict(list, clearesult or {})
        self.resultexport = export

    def _format_segment(self, r: dict, idx: int, store_results: bool = True) -> str:
        peak       = r.get('peak', float('nan'))
        t0         = r.get('t0', float('nan'))
        tau_w = r.get('tau_weighted', float('nan'))
        v0         = r.get('v_baseline', float('nan'))
        vp         = r.get('v_peak', float('nan'))
        dv         = r.get('delta_v', float('nan'))
        k0         = r.get('k_baseline', float('nan'))
        kp         = r.get('k_peak', float('nan'))
        dk         = r.get('delta_k', float('nan'))
        tau        = r.get('tau', float('nan'))
        lam        = r.get('lambda', float('nan'))
        r2         = r.get('r2', float('nan'))
        amps       = r.get('amps', [])
        Cval       = r.get('C', float('nan'))
        delay_time = r.get('decay_time', float('nan'))
        slice_x    = r.get('slice_x', [])
        slice_y    = r.get('slice_y', [])
        fit_x      = r.get('fit_x', [])
        fit_y      = r.get('fit_y', [])
        k90        = r.get('k90', float('nan'))
        k10        = r.get('k10', float('nan'))
        full_x     = r.get('full_x', [])
        full_y     = r.get('full_y', [])
        t_win      = r.get('t_win', [])
        y_win      = r.get('y_win', [])
        n_data     = r.get('n_data', float('nan'))
        slice_t0   = r.get('slice_t0', float('nan'))
        slice_dt   = r.get('slice_dt', float('nan'))
        seg_x      = r.get('seg_x', [])
        seg_y      = r.get('seg_y', [])
        fitted     = r.get('fitted', [])
    
        A_str = ", ".join(f"{a:.2f}" for a in amps) if amps else "—"
        v90 = v0 + 0.9 * dv
        v10 = v0 + 0.1 * dv
        # τ / λ in both unit systems
        tau_s  = r['tau']
        lam_s  = r['lambda']
        if isinstance(tau_s, (list, tuple)):
            tau_ms = [t * 1000 for t in tau_s]
            lam_ms = [l / 1000 for l in lam_s]

            tau_s_str  = ", ".join(f"{t:.2f}" for t in tau_s)
            tau_ms_str = ", ".join(f"{t:.1f}"  for t in tau_ms)
            lam_s_str  = ", ".join(f"{l:.3f}" for l in lam_s)
            lam_ms_str = ", ".join(f"{l:.6f}" for l in lam_ms)
        else:  # single-exponential (old behaviour)
            tau_ms     = tau_s * 1000
            lam_ms     = lam_s / 1000
            tau_s_str  = f"{tau_s:.2f}"
            tau_ms_str = f"{tau_ms:.1f}"
            lam_s_str  = f"{lam_s:.3f}"
            lam_ms_str = f"{lam_ms:.6f}"
        
        if store_results:
            self.clearance["peak"].append(peak)
            self.clearance["t0"].append(t0)
            self.clearance["v_baseline"].append(v0)
            self.clearance["v_peak"].append(vp)
            self.clearance["delta_v"].append(dv)
            self.clearance["k_baseline"].append(k0)
            self.clearance["k_peak"].append(kp)
            self.clearance["delta_k"].append(dk)
            self.clearance["tau"].append(tau)
            self.clearance["lambda"].append(lam)
            self.clearance["r2"].append(r2)
            self.clearance["amps"].append(amps)
            self.clearance["C"].append(Cval)
            self.clearance["decay_time"].append(delay_time)
            self.clearance["slice_x"].append(slice_x)
            self.clearance["slice_y"].append(slice_y)
            self.clearance["fit_x"].append(fit_x)
            self.clearance["fit_y"].append(fit_y)
            self.clearance["k90"].append(k90)
            self.clearance["k10"].append(k10)
            self.clearance["full_x"].append(full_x)
            self.clearance["full_y"].append(full_y)
            self.clearance["t_win"].append(t_win)
            self.clearance["y_win"].append(y_win)
            self.clearance["n_data"].append(n_data)
            self.clearance["slice_t0"].append(slice_t0)
            self.clearance["slice_dt"].append(slice_dt)
            self.clearance["seg_x"].append(seg_x)
            self.clearance["seg_y"].append(seg_y)
            self.clearance["fitted"].append(fitted)
            self.clearance["v90"].append(v90)
            self.clearance["v10"].append(v10)
            self.clearance["tau_s_str"].append(tau_ms_str)
            self.clearance["lam_s_str"].append(lam_ms_str)
            self.clearance["delay_time"].append(r['decay_time'])
            self.clearance["A_str"].append(A_str)
            self.clearance["tau_weighted"].append(tau_w)


        self.results["Analysis Type"].append(f"Segment {idx} ")
        self.results["Value"].append((
            f"\n    • V₀ (baseline)        : {v0:.2f} mV\n"
            f"    • Vₚ (peak)            : {vp:.2f} mV\n"
            f"    • ΔV (amplitude)       : {dv:.2f} mV\n"
            f"    • V₉₀ (90 %)           : {v90:.2f} mV\n"
            f"    • V₁₀ (10 %)           : {v10:.2f} mV\n"
            f"    • [K⁺]₀ (baseline)     : {k0:.2f} mM\n"
            f"    • [K⁺]ₚ (peak)         : {kp:.2f} mM\n"
            f"    • Δ[K⁺] (amplitude)    : {dk:.2f} mM\n"
            f"    • V₉₀ (mM)             : {k90:.2f} mM\n"
            f"    • V₁₀ (mM)             : {k10:.2f} mM\n"
            f"    • τ (Tau)              : {tau_s_str} s   ({tau_ms_str} ms)\n"
            f"    • λ = 1/τ              : {lam_s_str} s⁻¹   ({lam_ms_str} ms⁻¹)\n"
            f"    • τ𝚠 (weighted)         : {tau_w:.2f} s\n"
            f"    • Decay 90→10 %        : {r['decay_time']:.2f} s\n"
            f"    • R² (fit)             : {r['r2']:.3f}\n"
            f"    • A (mV)               : {A_str}\n"
            f"    • C (mV)               : {Cval:.2f}\n"))
        
        return (
            f"Segment {idx}\n"
            f"\t• V₀ (baseline)\t :{v0:.2f} mV\n"
            f"\t• Vₚ (peak)\t\t :{vp:.2f} mV\n"
            f"\t• ΔV (amplitude)\t :{dv:.2f} mV\n"
            f"\t• V₉₀ (90 %)\t\t :{v90:.2f} mV\n"
            f"\t• V₁₀ (10 %)\t\t :{v10:.2f} mV\n"
            f"\t• [K⁺]₀ (baseline)\t :{k0:.2f} mM\n"
            f"\t• [K⁺]ₚ (peak)\t\t :{kp:.2f} mM\n"
            f"\t• Δ[K⁺] (amplitude)\t :{dk:.2f} mM\n"
            f"\t• V₉₀ (mM)\t\t :{k90:.2f} mM\n"
            f"\t• V₁₀ (mM)\t\t :{k10:.2f} mM\n"
            f"\t• τ (Tau)\t\t :{tau_s_str} s   \t({tau_ms_str} ms)\n"
            f"\t• λ = 1/τ \t\t :{lam_s_str} s⁻¹  ({lam_ms_str} ms⁻¹)\n"
            f"\t• τ𝚠 (weighted)\t\t :{tau_w:.2f} s\n"
            f"\t• Decay 90→10 % \t :{r['decay_time']:.2f} s\n"
            f"\t• R² (fit)\t\t :{r['r2']:.3f}\n"
            f"\t• A (mV)\t\t :{A_str}\n"
            f"\t• C (mV)\t\t :{Cval:.2f}\n"
        )
        
    # Handles basic plotting functionality based on tags
    def handle_Plot(self, result):
        if isinstance(result, tuple):
            if isinstance(result[0], str):
                tag = result[0]

                if tag == "full_signal":
                    _, x, y, peaks = result
                    self.PlotArea.plot_abf_with_peaks(x, y, peaks)
                    self.logger.AppendMessage("[PROGRESS] Full signal plotted with peaks.")
                    self.results["Analysis Type"].append(f"Full signal plotted")
                    self.results["Value"].append(f"baseline {y.min():.2f}–{y.max():.2f} mM")
                    self.logger.AppendResults(f"Full signal plotted: baseline {y.min():.2f}–{y.max():.2f} mM")

                elif tag == "ABF":
                    _, path, x, y, current = result
                    self.PlotArea.set_file(path)          
                    self.PlotArea.plot_abf(x, y, current)
                    self.logger.AppendMessage("ABF loaded and plotted.")
                
                elif tag == "filtered":
                    _, mains, x, y = result
                    self.PlotArea.plot_filtered(x, y, mains)
                    self.logger.AppendMessage(f"[VALUE] {mains} Hz notch filter applied.")
                
                elif tag == "SegmentEcellular":
                    print(result)
                    _, seg = result
                    
                    for key in ["baseline", "high_k", "recovery"]:
                        if isinstance(seg[key], list) and len(seg[key]) == 1:
                            seg[key] = seg[key][0]
                            
                    print("Baseline 2nd", seg["baseline"])
                    print("High K 2nd", seg["high_k"])
                    print("Recovery 2nd", seg["recovery"])

                    baseline_time, baseline_signal = seg["baseline"]
                    highk_time, highk_signal = seg["high_k"]
                    recovery_time, recovery_signal = seg["recovery"]
                    
                    self.PlotArea.plot_ec_segments("Baseline", baseline_time, baseline_signal)
                    self.PlotArea.plot_ec_segments("High K⁺", highk_time, highk_signal)
                    self.PlotArea.plot_ec_segments("Recovery", recovery_time, recovery_signal)
                    
                    self.PlotArea.mark_ec_segments_on_raw(seg)
                    self.logger.AppendMessage("[PROGRESS] Segment applied")
                
                elif tag == "heatmap_trace":
                    _, heatresult = result
                    self.PlotArea.plot_heatmap(heatresult)
                    self.logger.AppendMessage("heatmap applied") 
                    
                elif tag == "spectral_density":
                    _, spectrumresult = result
                    self.PlotArea.plot_spectrum(spectrumresult)
                    self.logger.AppendMessage("power density applied")
                    
                elif tag == "zap_profile":
                    _, freq, impedance = result
                    self.PlotArea.plot_zap(freq, impedance)
                    self.logger.AppendMessage("zap profile applied.")
                
                elif tag == "clearance1":
                    _, cleartest, session = result
                    self.last_clearance = cleartest
                    
                    self.PlotArea.session = session
                    self.PlotArea.plot_clearance(cleartest)
                    
                    try:
                        self.PlotArea.fitUpdated.disconnect(self._on_fit_updated)
                    except TypeError:
                        pass
                    self.PlotArea.fitUpdated.connect(self._on_fit_updated)
                    
                    self.logger.AppendMessage("[PROGRESS] All clearance applied.")

            else:
                x, y , current  = result
                self.PlotArea.plot_abf(x, y, current)
                self.logger.AppendMessage("[PROGRESS] ABF loaded and plotted.")
        else:
            self.logger.AppendMessage(str(result))
    
    # Resets all display graphs
    def reset_display(self):
        """Nuke plots + text when a new file is chosen."""
        self.PlotArea.reset()
        self.logger.clear_all()
        self.last_clearance        = None
        self.PlotArea.session      = None
        self.PlotArea.current_results = []
        self.PlotArea.comparison_results = None
        
    def printAnalysisType(self, Tabresult):
        print(f"[DEBUG] file_path type: {type(Tabresult)} - {Tabresult}")
        print(f"[DEBUG] isinstance(Tabresult, str): {isinstance(Tabresult, str)}")
        if isinstance(Tabresult, str):
            self.logger.AppendMessage(f"[INFO] Detected Analysis Type: {Tabresult}")
            return Tabresult
        else:
            self.logger.AppendMessage("[WARNING] Could not determine analysis type")
            return None
    # Extracellular Result handles
    
    # Handles Extracellular Linear Region functionality
    def handle_RegionPlotting(self):
        SegmentationState = self.PlotArea.SegmentationRegion()
        if SegmentationState == True:
            self.logger.AppendMessage("[PROGRESS] Segmentation Regions Ready and Plotted")
            return SegmentationState
        else:
            self.logger.AppendMessage("[ERROR] Unable to show Regions, No ABF file Loaded")
            return SegmentationState
        
    # Handles removal of Linear Regions 
    def handle_RemoveRegion(self):
        RemoveSegment = self.PlotArea.RemoveSegmentationRegion()
        
        if RemoveSegment == True:
            self.logger.AppendMessage("[PROGRESS] Segmentation Region and Plot Removed")
            return RemoveSegment
        else:
            self.logger.AppendMessage("[ERROR] No Segments to remove")
            return RemoveSegment
    
    # Handle region data fetching
    def handle_dataFetch(self):
        segments, ErrorState = self.PlotArea.FetchECellularData()
        if segments is None and ErrorState:
            self.logger.AppendMessage("[ERROR] No Segments Found")
            self.baselineData = None
            self.highkData = None
            self.recoveryData = None
            
            return self.baselineData, self.highkData, self.recoveryData
        elif ErrorState:
            self.logger.AppendMessage("[ERROR] Segmentation hasn't been done")
            self.baselineData = None
            self.highkData = None
            self.recoveryData = None
            
        else:
            self.logger.AppendMessage("[PROGRESS] Segments have been fetched")
            
            self.baselineData = segments["baseline"]
            self.highkData = segments["high_k"]
            self.recoveryData = segments["recovery"]
            
            return self.baselineData, self.highkData, self.recoveryData
    
    # Handles the segmenting
    def handle_ECsegement(self, result):
        if isinstance(result, dict):
            try:
                baseline_time, baseline_signal = result["baseline"]
                highk_time, highk_signal = result["high_k"]
                recovery_time, recovery_signal = result["recovery"]
                
                self.segment["Baseline"].append(result["baseline"])
                self.segment["HighK"].append(result["high_k"])
                self.segment["Recovery"].append(result["recovery"])

                # Plot each segment individually using your plot widget
                self.PlotArea.plot_ec_segments("Baseline", baseline_time, baseline_signal)
                self.PlotArea.plot_ec_segments("High K⁺", highk_time, highk_signal)
                self.PlotArea.plot_ec_segments("Recovery", recovery_time, recovery_signal)

                self.PlotArea.mark_ec_segments_on_raw(result)

                self.logger.AppendMessage("[PROGRESS] Baseline, High K⁺, and Recovery segments plotted.")
            except Exception as e:
                self.logger.AppendMessage(f"[ERROR]: Failed to unpack segment data: {str(e)}")
                traceback.print_exc()
        else:
            self.logger.AppendMessage("[ERROR]: Segment result is not a dictionary")
            
    # Handles the removal of the extracted data
    def handle_DataRemoval(self):
        labels = {"Baseline Segment", "High K⁺ Segment", "Recovery Segment"}
        UnmarkedState = self.PlotArea.Unmark_segments(labels)
        if UnmarkedState:
            self.baselineData = None
            self.highkData = None
            self.recoveryData = None
            self.logger.AppendMessage("[PROGRESS] Extracted Data has been removed")
        else: 
            self.logger.AppendMessage("[ERROR] Extracted Data does not exist")
            
        return self.baselineData, self.highkData, self.recoveryData

    # Handles the data for heatmap Plotting
    def handle_HeatmapPlot(self, result):
        if isinstance(result, dict):
            heatmap_data = {}
            
            # Add section header
            self.logger.AppendResults("Heatmap Analysis")
            self.results["Analysis Type"].append("Heatmap Analysis")
            self.results["Value"].append("")
            
            for label, (f, t, Sxx_db, freq, time, power, Spec_min, Spec_max) in result.items():
                # Only pass what plot_heatmap expects
                heatmap_data[label] = (f, t, Sxx_db, Spec_min, Spec_max)
                self.heatmap["labels"].append(label)
                self.heatmap["f"].append(f)
                self.heatmap["t"].append(t)
                self.heatmap["Sxx"].append(Sxx_db)
                self.heatmap["freq"].append(freq)
                self.heatmap["time"].append(time)
                self.heatmap["power"].append(power)
                self.heatmap["Spec_min"].append(Spec_min)
                self.heatmap["Spec_max"].append(Spec_max)
                self.results["Analysis Type"].append(f"\t{label}: ")
                self.results["Value"].append(f"\tPeak Power \t{power:.2f} (dB) at {freq:.2f} Hz, {time:.2f} s")

                analysis_line = f"\t{label + ':':<10}\tPeak Power \t{power:.2f} (dB) at {freq:.2f} (Hz), {time:.2f} s"
                self.logger.AppendResults(analysis_line)
                
            self.PlotArea.plot_heatmap(heatmap_data)
            self.logger.AppendMessage("[PROGRESS] Segment heatmaps have been plotted.")
        else:
            self.logger.AppendMessage("[ERROR] Plot is not loaded, Open File and segment to create Heatmaps")
    
    # Handles the plotting of the Power Spectral Density       
    def handle_SpectrumPlot(self, result):
        if isinstance(result, dict):
            # Check expected structure
            required_keys = {"baseline", "high_k", "recovery"}
            if not required_keys.issubset(result.keys()):
                self.logger.AppendMessage("[ERROR] Missing one or more required spectrum keys.")
                return
            
            self.logger.AppendResults("\nSpectrum Analysis")
            self.results["Analysis Type"].append("Spectrum Analysis")
            self.results["Value"].append("")
                        
            # Plot all spectrum data
            self.PlotArea.plot_spectrum(result)

            # Log max PSDs for each condition
            for label in ["baseline", "high_k", "recovery"]:
                print(label)
                self.spectrum["labels"].append(label)
                self.spectrum["freqs"].append(result[label]["freqs"])
                self.spectrum["mean_psd"].append(result[label]["mean"])
                self.spectrum["sem_psd"].append(result[label]["sem"])
                
                max_psd = result[label]["mean"].max()
                self.results["Analysis Type"].append(f"Max PSD: {label} = ")
                self.results["Value"].append(f"{max_psd:.4f} (µV² × 10⁻²)")

                self.logger.AppendResults(f"\tMax PSD:\t{label:<8} = {max_psd:.4f} (µV² × 10⁻²)")
                
            self.logger.AppendMessage("PROGRESS: Power Spectral Density plotted for all segments.")

        else:
            self.logger.AppendMessage("[ERROR] Plot is not Loaded, Open File and segment to create Spectral Density")
    
    # Resonance frequency Result Handles 

    # Handles the plotting of the Impedance Plot
    def handle_ImpedancePlot(self, result):
        if isinstance(result, tuple):
            freq, impedance = result
            self.PlotArea.plot_zap(freq, impedance)
            self.logger.AppendMessage("[PROGRESS] Impedance Plot has been plotted")
            
            self.zap["Frequency"].append(freq)
            self.zap["Impedance"].append(impedance)
        else:
            self.logger.AppendMessage("[ERROR] Plot is not loading")
    
    # Handles the results of the calculation of the Resonance Frequency
    def handle_CalcResonanceFreq(self, result):
        if isinstance(result, float):

            self.logger.AppendResults("Resonance Analysis")
            self.logger.AppendResults(f"\tResonance Frequency:\t{result:.2f} Hz")
            self.logger.AppendMessage("PROGRESS: Resonance Frequency has been shown")

            
            self.results["Analysis Type"].append("Resonance Frequency")
            self.results["Value"].append(f"{result:.2f} Hz")
        else:
            self.logger.AppendMessage("[ERROR] Results is not logging properly")
    
    # Handles the results of the calculation of the Half-Width at Attenaution
    def handle_CalcHWA(self, result):
        if isinstance(result, tuple):
            f_left, f_right, z_left, z_right, HWA = result
            self.logger.AppendResults(f"\tHalf-Width Attenuation: \t{HWA:.2f} Hz")
            self.PlotArea.add_point(f_left, z_left)
            self.PlotArea.add_point(f_right, z_right)
            self.logger.AppendMessage("[PROGRESS] Half-Width at Attenuation has been shown")
            
            self.results["Analysis Type"].append("Half-Width Attenuation")
            self.results["Value"].append(HWA)
        else:
            self.logger.AppendMessage("[ERROR] Results is not logging properly")
    
    # K+ Clearance result handles 
    
    # Handles the K+ analyses Results 
    def handle_Clearance_results(self, payload):
        """
        Slot connected to the worker's 'result' signal when the task is
        'SegmentEpisodes'.  *payload* is a 3-tuple:
            ("clearance", results_list, ClearanceSession)
        """
        # tag, segments, session = payload
        # if tag != "clearance":
        #     return     
        # 
        if not isinstance(payload, tuple) or len(payload) < 2:
            return   

        tag, segments, *rest = payload
        if tag != "clearance":
            return

        session = rest[0] if rest else None                 

        self.last_clearance = segments

        if hasattr(self.logger, 'clear_results'):
            self.logger.clear_results()
        else:
            self.logger.clear_all()

        for i, seg in enumerate(segments, 1):
            self.logger.AppendResults(self._format_segment(seg, i))

        self.PlotArea.session = session
        self.PlotArea.plot_clearance(segments)

        # (re)connect the live-update signal
        try:
            self.PlotArea.fitUpdated.disconnect(self._on_fit_updated)
        except TypeError:
            pass
        self.PlotArea.fitUpdated.connect(self._on_fit_updated)

        self.logger.AppendMessage("PROGRESS: Episode segmented!")

    '''# Handles the formatting of results for K+ Clearance
    def _format_segment(self, r: dict, idx: int) -> str:
        v0  = r.get('v_baseline', float('nan'))
        vp  = r.get('v_peak',     float('nan'))
        dv  = r.get('delta_v',    float('nan'))
        k0  = r.get('k_baseline', float('nan'))
        kp  = r.get('k_peak',     float('nan'))
        dk  = r.get('delta_k',    float('nan'))
        k90 = r.get('k90',        float('nan'))
        k10 = r.get('k10',        float('nan'))
        T3 = r.get('T3', float('nan'))
        amps = r.get('amps', [])
        Cval = r.get('C', float('nan'))
        A_str = ", ".join(f"{a:.2f}" for a in amps) if amps else "—"
        v90 = v0 + 0.9 * dv
        v10 = v0 + 0.1 * dv
        # τ / λ in both unit systems
        tau_s  = r['tau']
        lam_s  = r['lambda']
        if isinstance(tau_s, (list, tuple)):
            tau_ms = [t * 1000 for t in tau_s]
            lam_ms = [l / 1000 for l in lam_s]

            tau_s_str  = ", ".join(f"{t:.2f}" for t in tau_s)
            tau_ms_str = ", ".join(f"{t:.1f}"  for t in tau_ms)
            lam_s_str  = ", ".join(f"{l:.3f}" for l in lam_s)
            lam_ms_str = ", ".join(f"{l:.6f}" for l in lam_ms)
        else:  # single-exponential (old behaviour)
            tau_ms     = tau_s * 1000
            lam_ms     = lam_s / 1000
            tau_s_str  = f"{tau_s:.2f}"
            tau_ms_str = f"{tau_ms:.1f}"
            lam_s_str  = f"{lam_s:.3f}"
            lam_ms_str = f"{lam_ms:.6f}"
        return (
            f"Segment {idx}\n"
            f"\t• V₀ (baseline)\t:{v0:.2f}\tmV\n"
            f"\t• Vₚ (peak)\t\t:{vp:.2f}\tmV\n"
            f"\t• ΔV (amplitude)\t:{dv:.2f}\tmV\n"
            f"\t• V₉₀ (90 %)\t\t:{v90:.2f}\tmV\n"
            f"\t• V₁₀ (10 %)\t\t:{v10:.2f}\tmV\n"
            f"\t• [K⁺]₀ (baseline)\t:{k0:.2f}\tmM\n"
            f"\t• [K⁺]ₚ (peak)\t\t:{kp:.2f}\tmM\n"
            f"\t• Δ[K⁺] (amplitude)\t:{dk:.2f}\tmM\n"
            f"\t• V₉₀ (mM)\t\t:{k90:.2f}\tmM\n"
            f"\t• V₁₀ (mM)\t\t:{k10:.2f}\tmM\n"
            f"\t• τ (Tau)\t\t:{tau_s:.2f}\ts\t({tau_ms:.1f} ms)\n"
            f"\t• λ = 1/τ\t\t:{lam_s:.3f}s⁻¹\t({lam_ms:.6f} ms⁻¹)\n"
            f"\t• Decay 90→10 %\t:{r['decay_time']:.2f} s\n"
            f"\t• R² (fit)\t\t:{r['r2']:.3f}"
        )'''

    # Handles the export results for clearance
    def export_clearance(self, dir_path: str, image_fmt="png"):
        """
        Write CSV + plots to *dir_path* (must already exist).
        """
        if not self.last_clearance:
            QMessageBox.warning(
                None, "No data",
                "Run 'Segment Episodes' first; there are no clearance "
                "results to export yet."
            )
            return

        csv_path  = results_to_csv(self.last_clearance, dir_path)
        img_paths = export_clearance_plots(self.PlotArea, dir_path, image_fmt)

        msg = (f"CSV saved → {csv_path}\n"
            f"{len(img_paths)} plot(s) saved as {image_fmt.upper()}")
        self.logger.AppendMessage(msg)
        QMessageBox.information(None, "Export complete", msg)

    def _on_fit_updated(self, seg_idx: int, res: dict):
        """Update cached list and refresh ENTIRE results pane."""
        if self.last_clearance is None:
            return  # should never happen
        # overwrite the dict for this segment (1‑based index comes in)
        self.last_clearance[seg_idx - 1].update(res)

        # ---------- full refresh ----------
        if hasattr(self.logger, 'clear_results'):
            self.logger.clear_results()
        else:
            self.logger.clear_all()
        for i, seg in enumerate(self.last_clearance, 1):
            self.logger.AppendResults(self._format_segment(seg, i))

        self.logger.AppendMessage(f"[PROGRESS]Segment {seg_idx} parameters updated.")