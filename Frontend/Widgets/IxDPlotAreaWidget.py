import os
import pyqtgraph as pg
import matplotlib.pyplot as plt
from pyqtgraph import PlotWidget
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from pyqtgraph import TextItem 
from scipy.optimize import curve_fit
from Backend.Services.clearance import ClearanceSession

MAX_FIT_AFTER_90_SEC = 15.0

class IxDPlotArea(QWidget):
    fitUpdated = pyqtSignal(int, dict)
    def __init__(self, session: ClearanceSession | None = None):
        super().__init__()
        self.session = session 
        self.current_results = []
        
        self.x_data = None
        self.y_data = None
        self.ABFloaded = False
        
        self.start_time = 2
        self.end_time = 4
        
        # Main vertical layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Tab widget to hold the Raw ABF plot plus one tab per clearance segment
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.raw_tab = QWidget()
        self.raw_layout = QVBoxLayout()
        self.raw_tab.setLayout(self.raw_layout)
        
        pg.setConfigOption('background', 'w')
        
        # Raw ABF tab (always present)
        self.channel0_Plot = pg.PlotWidget()
        self.channel0_Plot.setTitle("Raw ABF - Channel 0")
        self.raw_layout.addWidget(self.channel0_Plot)

        self.raw_plot = self.channel0_Plot
        
        # Current plot (optional)
        self.channel1_Plot = pg.PlotWidget()
        self.channel1_Plot.setTitle("Raw ABF - Channel 1")
        self.channel1_Plot_visible = False
        self.comparison_results = None
        self.tabs.addTab(self.raw_tab, "Raw ABF")

    def _tau_lambda_strings(self, r):
        """
        Return (tau_str, lam_str) ready for f-strings, whether the
        values are single floats or lists/tuples.
        """
        tau = r["tau"]
        lam = r["lambda"]

        if isinstance(tau, (list, tuple)):
            tau_str = ", ".join(f"{t:.2f}" for t in tau)
            lam_str = ", ".join(f"{l:.3f}" for l in lam)
        else:
            tau_str = f"{tau:.2f}"
            lam_str = f"{lam:.3f}"
        return tau_str, lam_str
    
    # Raw ABF Plotting
    def plot_abf(self, x, y, current= None):
        """Update the Raw ABF tab."""
        title = (f"{getattr(self, 'file_name', 'Raw ABF')}  –  Channel 0")
        self.x_data = x
        self.y_data = y
        
        self.ABFloaded = True
        self.channel0_Plot.clear()

        self.channel0_Plot.clear()
        self.channel0_Plot.setTitle(title)
        self.channel0_Plot.plot(x, y, pen=pg.mkPen('Black', width=1))
        self.channel0_Plot.setLabel('left', 'K+ Level(Potassium)')
        self.channel0_Plot.setLabel('bottom', 'Time (S)')
        self.channel0_Plot.showGrid(x=True, y=True)
        
        if current is not None:
            self.channel1_Plot.clear()
            self.channel0_Plot.setLabel('left', 'Voltage (mV)')
            self.channel0_Plot.setLabel('bottom', 'Time (S)')
            self.channel1_Plot.plot(x, current, pen=pg.mkPen('Black', width=1), name="Current (pA)")
            self.channel1_Plot.setLabel('left', 'Current (pA)')
            self.channel1_Plot.setLabel('bottom', 'Time (S)')
            self.channel1_Plot.showGrid(x=True, y=True)
            if not self.channel1_Plot_visible:
                self.raw_layout.addWidget(self.channel1_Plot)
                self.channel1_Plot_visible = True
        else: 
            if self.channel1_Plot_visible:
                self.raw_layout.removeWidget(self.channel1_Plot)
                self.channel1_Plot.setParent(None)
                self.channel1_Plot_visible = False
                
    # Resets the raw ABF file
    def reset(self):
        """
        Remove all segment tabs and wipe the raw‐ABF plot back to blank.
        """
        # 1) kill all tabs except the first ("Raw ABF")
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)

        # 2) clear the raw‐data plot
        self.channel0_Plot.clear()
        self.channel0_Plot.showGrid(x=True, y=True)

        # 3) restore your original title & axis labels
        self.channel0_Plot.setTitle("Raw ABF – Channel 0")
        self.channel0_Plot.setLabel('left',  'Voltage (mV)')
        self.channel0_Plot.setLabel('bottom','Time (s)')
    
    # Extracellular Plot related functions 
    
    # Segmenting Regions
    def SegmentationRegion(self):
        SegmentationState = False
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Segment":
                self.tabs.removeTab(i)
                break
        
        if self.ABFloaded == True:
            self.SegmentationPlot = pg.PlotWidget()
            self.SegmentationPlot.setTitle("Segmentation Plot")
            self.SegmentationPlot.plot(self.x_data, self.y_data, pen=pg.mkPen('Black', width=1))
            self.SegmentationPlot.showGrid(x=True, y=True)
            
            region_colors = {
                "baseline": (0, 0, 255, 80),   # blue with alpha
                "high_k": (255, 0, 0, 80),     # red with alpha
                "recovery": (0, 255, 0, 80),   # green with alpha
            }
            
            # Sets a linear region on raw abf to extract data.
            self.baselineRegion = pg.LinearRegionItem(values=(self.start_time, self.end_time), brush=(region_colors["baseline"]), pen="b")
            self.SegmentationPlot.addItem(self.baselineRegion)
            
            self.HighKRegion = pg.LinearRegionItem(values=(self.start_time, self.end_time), brush=(region_colors["high_k"]), pen="r")
            self.SegmentationPlot.addItem(self.HighKRegion)
            
            self.RecoveryRegion = pg.LinearRegionItem(values=(self.start_time, self.end_time), brush=(region_colors["recovery"]), pen="g")
            self.SegmentationPlot.addItem(self.RecoveryRegion)
            
            self.tabs.addTab(self.SegmentationPlot, "Segment")
            self.tabs.setCurrentWidget(self.SegmentationPlot)
            SegmentationState = True
            return SegmentationState
        else:
            print("Nothing to segment")
            SegmentationState = False
            return SegmentationState
        
    # Removes the Segmentation regions
    def RemoveSegmentationRegion(self):
        SegmentationState = False
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Segment":
                self.tabs.removeTab(i)
                SegmentationState = True
                break
        # Removes Linear Regions
        self.SegmentationPlot.removeItem(self.baselineRegion)
        self.SegmentationPlot.removeItem(self.HighKRegion)
        self.SegmentationPlot.removeItem(self.RecoveryRegion)
        
        # Completely delete the region objects
        self.baselineRegion = None
        self.HighKRegion = None
        self.RecoveryRegion = None
        
        return SegmentationState
        
    # Fetches the segmented data
    def FetchECellularData(self):
        ErrorState = False
        if self.x_data is None or self.y_data is None:
            ErrorState = True
            return None, ErrorState
        else:
            x = self.x_data
            y = self.y_data
        
            baseline = (np.array([]), np.array([]))
            highk = (np.array([]), np.array([]))
            recovery = (np.array([]), np.array([]))

            try:
                baseline_start, baseline_end = self.baselineRegion.getRegion()
                if baseline_start == baseline_end:
                    raise ValueError("Baseline region not defined.")
                idx0 = np.searchsorted(x, baseline_start)
                idx1 = np.searchsorted(x, baseline_end)
                baseline = (x[idx0:idx1], y[idx0:idx1])
            except Exception as e:
                print(f"[ERROR] Baseline segment error: {e}")
                ErrorState = True
                baseline = None

            try:
                highk_start, highk_end = self.HighKRegion.getRegion()
                if highk_start == highk_end:
                    raise ValueError("Baseline region not defined.")
                idx0 = np.searchsorted(x, highk_start)
                idx1 = np.searchsorted(x, highk_end)
                highk = (x[idx0:idx1], y[idx0:idx1])
            except Exception as e:
                print(f"[ERROR] High K+ segment error: {e}")
                ErrorState = True
                highk = None

            try:
                rec_start, rec_end = self.RecoveryRegion.getRegion()
                if rec_start == rec_end:
                    raise ValueError("Baseline region not defined.")
                idx0 = np.searchsorted(x, rec_start)
                idx1 = np.searchsorted(x, rec_end)
                recovery = (x[idx0:idx1], y[idx0:idx1])
            except Exception as e:
                print(f"[ERROR] Recovery segment error: {e}")
                ErrorState = True
                recovery = None

            return {"baseline": baseline, 
                    "high_k": highk, 
                    "recovery": recovery} , ErrorState
        
    # Marks out Segments
    def mark_ec_segments_on_raw(self, segments):
        print("[DEBUG] Entered mark_ec_segments_on_raw")

        # Remove previous shaded regions
        for item in self.SegmentationPlot.items():
            if isinstance(item, pg.LinearRegionItem):
                self.SegmentationPlot.removeItem(item)

        # Remove any previous dummy legend items
        for item in self.SegmentationPlot.listDataItems():
            if hasattr(item, 'isDummy') and item.isDummy:
                self.SegmentationPlot.removeItem(item)

        # Define region colors
        region_colors = {
            "baseline": (0, 0, 255, 80),   # blue with alpha
            "high_k": (255, 0, 0, 80),     # red with alpha
            "recovery": (0, 255, 0, 80),   # green with alpha
        }
        
        self.SegRegions = {}

        # Add shaded regions
        for label in ["baseline", "high_k", "recovery"]:
            segment = segments.get(label)
            if not segment or not isinstance(segment, tuple) or len(segment) != 2:
                print(f"[WARN] Skipping '{label}' – invalid or missing segment.")
                continue

            t_array, _ = segment
            if not isinstance(t_array, np.ndarray) or len(t_array) < 2:
                print(f"[WARN] Skipping '{label}' – time array missing or too short.")
                continue
            
            try:
                start = float(t_array[0])
                end = float(t_array[-1])
                region = pg.LinearRegionItem(values=(start, end), brush=region_colors[label])
                region.setMovable(False)
                region.setZValue(10)
                self.SegmentationPlot.addItem(region)
                self.SegRegions[label] = region
                print(f"[DEBUG] Added region for '{label}': {start:.2f} to {end:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to draw region for '{label}': {e}")
            
            print(self.SegRegions)

        # Add dummy items to simulate a legend
        legend = self.SegmentationPlot.addLegend(offset=(10, 10))

        def add_dummy_legend(name, color):
            dummy = self.SegmentationPlot.plot([0], [0], pen=pg.mkPen(color=color, width=8), name=name)
            dummy.isDummy = True
            dummy.setVisible(False)  # Hide line but show in legend

        add_dummy_legend("Baseline", (0, 0, 255))
        add_dummy_legend("High K⁺", (255, 0, 0))
        add_dummy_legend("Recovery", (0, 255, 0))
        
    def Unmark_segments(self, labels):
        UnmarkedState = False
        if hasattr(self, 'SegRegions'):
            for region in self.SegRegions.values():
                self.SegmentationPlot.removeItem(region)
                
           
            # Remove all matching tabs
            if isinstance(labels, (set, list, tuple)):
                for label in labels:
                    for i in reversed(range(self.tabs.count())):
                        if self.tabs.tabText(i) == label:
                            self.tabs.removeTab(i)
                            print(f"Tab '{label}' Removed")
            else:
                for i in reversed(range(self.tabs.count())):
                    if self.tabs.tabText(i) == labels:
                        self.tabs.removeTab(i)
                        print(f"Tab '{labels}' Removed")
            
            self.baselineRegion = None
            self.HighKRegion = None
            self.RecoveryRegion = None
                
            self.SegRegions.clear()
            
            UnmarkedState = True
            print(UnmarkedState)
            return UnmarkedState
        else:
            print("[ERROR] Can't Unmark the segments")
            UnmarkedState = False
            return UnmarkedState

     # Plots the specific Segments
    def plot_ec_segments(self, label, time_array, signal_array):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == label:
                self.tabs.removeTab(i)
                break
            
        # Create a new plot widget
        ec_plot = pg.PlotWidget()
        ec_plot.setTitle(f"Extracellular Segment - {label}")
        ec_plot.setLabel('bottom', 'Time (s)')
        ec_plot.setLabel('left', 'Voltage (mV)')
        ec_plot.showGrid(x=True, y=True)

        # Plot the segment
        ec_plot.plot(time_array, signal_array, pen=pg.mkPen('black', width=1))

        # Add to a new tab
        self.tabs.addTab(ec_plot, f"{label} Segment")
        self.comparison_results = None
    
    # Plots heatmap
    def plot_heatmap(self, heatmap_dict):
        if not heatmap_dict:
            print("No heatmap data to plot.")
            return
        
        for label, (f, t, Sxx_db, Spec_min, Spec_max, *_) in heatmap_dict.items():
            # Apply frequency mask: 0–100 Hz
            freq_mask = (f >= 0) & (f <= 100)
            f_limited = f[freq_mask]
            Sxx_db_limited = Sxx_db[freq_mask, :]
            # Calls the globally set minimum and maximum
            global_min = Spec_min
            global_max = Spec_max
            
            print(global_min, global_max)

            if f_limited.size == 0 or Sxx_db_limited.size == 0:
                print(f"Skipping {label}: no data in 0–100 Hz")
                continue
            
            # Creates heatmap widget
            heatmap_plot = pg.PlotWidget()

            # Sxx_db shape: (frequency bins, time bins)
            img = pg.ImageItem(Sxx_db_limited.T)

            # Set correct position and scaling
            img.setRect(pg.QtCore.QRectF(
                t[0], f_limited[0], t[-1] - t[0], f_limited[-1] - f_limited[0]
            ))
            
            colormap = pg.colormap.get('jet', source='matplotlib')
            lut = colormap.getLookupTable(0.0, 1.0, 256)
            img.setLookupTable(lut)
            img.setLevels((global_min, global_max))
            
            # Add the image
            heatmap_plot.addItem(img)
            heatmap_plot.setLabel('left', 'Frequency (Hz)')
            heatmap_plot.setLabel('bottom', 'Time (s)')
            heatmap_plot.setTitle(f"Segment Heatmap: {label}")
            heatmap_plot.setXRange(t[0], t[-1], padding=0)

            # Add to tab widget
            self.tabs.addTab(heatmap_plot, f"{label.capitalize()} Heatmap")
    
    # Plots Powers Spectral Density
    def plot_spectrum(self, spectrum_data):
        spectrum_plot = pg.PlotWidget()

        # Legend
        if not spectrum_plot.plotItem.legend:
            spectrum_plot.addLegend(offset=(500, 20))

        # Sets Colours
        colors = {
            "baseline": ('b', (0, 0, 255, 100)),
            "high_k": ('r', (255, 0, 0, 100)),
            "recovery": ('g', (0, 255, 0, 100)),
        }

        for label, data in spectrum_data.items():
            freqs = data["freqs"]
            mean_psd = data["mean"]
            sem_psd = data["sem"]

            color_line, color_fill = colors.get(label, ('k', (255, 255, 255, 100)))

            # Plot mean line
            curve = pg.PlotCurveItem(freqs, mean_psd, pen=pg.mkPen(color_line, width=2), name=label)
            spectrum_plot.addItem(curve)

            # Plot shaded SEM band
            upper = mean_psd + sem_psd
            lower = mean_psd - sem_psd
            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(freqs, upper),
                pg.PlotCurveItem(freqs, lower),
                brush=pg.mkBrush(color_fill)
            )
            spectrum_plot.addItem(fill)

        spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        spectrum_plot.setLabel('left', 'Spectral Power Density (μV² × 10⁻²)')
        spectrum_plot.showGrid(x=True, y=True)
        spectrum_plot.enableAutoRange()

        self.tabs.addTab(spectrum_plot, "Spectrum")
    
    # Resonance Frequency related Plots and functions
    
    # Adds the HWA points to the plot
    def add_point(self, x, y):
        self.zap_plot.plot([x], [y], pen=None, symbol='o', symbolsize=10, symbolbrush='r')
    
    # Plots the zap Profile
    def plot_zap(self, freq, impedance):
        """Plot on new zap profile tab."""
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)
            
        if freq is None or impedance is None or len(freq) == 0:
            print("[ERROR] Nothing to plot: freq or impedance is empty.")
            return
        
        self.zap_plot = pg.PlotWidget()
        self.zap_plot.setTitle("Impedance Plot")
        self.zap_plot.setLabel("bottom", "Frequency (Hz)")
        self.zap_plot.setLabel("left", "Impedance (MΩ)")
        self.zap_plot.showGrid(x=True, y=True)
        self.zap_plot.plot(freq, impedance, pen='Black')
        
        self.tabs.addTab(self.zap_plot, f"Zap Profile")

    # K+ Clearance related Plots and Functions
    
    # Plots the K+ Clearance analysis fit
    def plot_clearance(self, results):
        """
        Create one tab per clearance segment.  Stores an immutable copy of the
        *initial* fit so the slider can always restore it later.
        """
        # wipe old segment tabs (keep “Raw ABF” at index 0)
        while self.tabs.count() > 1:
            self.tabs.removeTab(1)

        self.current_results = results  # global cache for slider callback
        
        print("\n")
        print("See me now", results)
        print("\n")

        for i, r in enumerate(results, 1):
            pw = pg.PlotWidget()
            pw.addLegend(offset=(-100, 10))
            pw.setTitle(f"Segment #{i}")

            pk_time = r["t0"]
            pre_start = pk_time - 1.0
            mask_pre = (self.session.x >= pre_start) & (self.session.x <= pk_time)
            pre_x = self.session.x[mask_pre] - pk_time
            # pre_y = self.session.conc[mask_pre]
            raw_y = (self.session.filtered if
                     self.session.fit_domain == "voltage" else
                     self.session.conc)
            pre_y = raw_y[mask_pre]

            pw.plot(pre_x, pre_y,
                    pen=pg.mkPen(width=1, color=(150,150,150)),
                    name="pre-peak")

            # ── 1 · context window (±30 s) ────────────────────────────────
            pw.plot(r["t_win"], r["y_win"],
                    pen=pg.mkPen(width=1, color=(120, 120, 120)),
                    name="context")

            # offset so the slice sits on the correct absolute time-axis
            # offset = r["t_win"][np.where(r["y_win"] == r["slice_y"][0])[0][0]]
            # slice_abs_x = r["slice_x"] + offset


            # ── 2 · initial raw slice + initial fit ───────────────────────
            # raw_curve = pw.plot(slice_abs_x, r["slice_y"], pen='k', name="slice")
            # raw_curve._tag = "raw"

            # fit_curve = pw.plot(r["fit_x"] + offset, r["fit_y"],
                                # pen='r', name="fit")
            # fit_curve._tag = "fit"

            # --- 1 · slice + fit on the same axis as t_win --------------------
            offset   = r["slice_dt"]            # seconds after peak
            slice_x  = r["seg_x"] + offset      # seg_x starts at 0

            raw_curve = pw.plot(
                slice_x, r["seg_y"],
                pen=pg.mkPen('k', width=1), name="slice")
            raw_curve._tag = "raw"

            n_pts = r["n_data"]              
            fit_curve = pw.plot(
                slice_x[:n_pts],             # X up to the last data point
                r["fitted"][:n_pts],         # same number of Y points
                pen=pg.mkPen('r'), width=1,
                name="fit"
            )
            fit_curve._tag = "fit"

            # ── 3 · freeze “factory” values for this segment ──────────────
            r.update({
                "init_bounds" : (slice_x[0], slice_x[-1]),
                "init_slice_x": slice_x.copy(),
                "init_slice_y": r["seg_y"].copy(),
                "init_tau"    : r["tau"],
                "init_r2"     : r["r2"],
                "init_fit"    : r["fitted"].copy(),
                "last_bounds" : (slice_x[0], slice_x[-1]),
                "last_fit"    : r["fitted"].copy(),
            })

            # ── 4 · draggable LinearRegionItem ────────────────────────────
            region = pg.LinearRegionItem(values=list(r["init_bounds"]),
                                        brush=(50, 50, 200, 30))
            pw.addItem(region)
            region.sigRegionChangeFinished.connect(
                lambda reg=region, idx=i-1, plot=pw:
                self._on_region_update(reg, idx, plot)
            )

            # ── 5 · k90 / k10 guide lines — now DRAGGABLE ─────────────────────────
            for y_val, (col, tag) in (
                    (r["k90"], ((0, 0, 160), "k90_line")),
                    (r["k10"], ('c', "k10_line"))
            ):
                line = pg.InfiniteLine(
                    pos=y_val, angle=0, movable=True,
                    pen=pg.mkPen(color=col, style=Qt.DashLine)
                )
                line._tag         = tag       # “k90_line” or “k10_line”
                line._seg_idx     = i - 1     # result‑array index
                line._plot_widget = pw        # ← store the PlotWidget
                pw.addItem(line)
                line.sigPositionChangeFinished.connect(
                    lambda ln=line: self._on_threshold_update(ln)
                )

            # ── 6 · τ / R² label ─────────────────────────────────────────
            tau_str, lam_str = self._tau_lambda_strings(r)
            tw_str = f"{r['tau_weighted']:.2f}"
            text = (
                f"τ={tau_str}s\n"
                f"R²={r['r2']:.3f}\n"
                f"λ={lam_str}s⁻¹\n"
                f"τ𝚠={tw_str}s"           # amplitude-weighted τ
            )

            label = pg.TextItem(text, color=(220, 0, 0), anchor=(0, 1))
            pw.addItem(label)
            label.setPos(slice_x[0], r["seg_y"][0] * 1.05)
            label._tag = "lbl"


            self.tabs.addTab(pw, f"Seg {i}")

    # Plots the ABF file with the peaks shown
    def plot_abf_with_peaks(self, x, y, peaks):
        """
        Plot the full [K+] sweep and overlay red peak markers.
        """
        self.channel0_Plot.clear()
        self.channel0_Plot.plot(x, y, pen='Black')
        if len(peaks):
            self.channel0_Plot.plot(
                x[peaks], y[peaks],
                pen=None,
                symbol='o',
                symbolBrush='r',
                symbolSize=6
            )

    # Plots the filtered K+ Plots. Supports both 50hz and 60hz
    def plot_filtered(self, x, y, mains):
        """
        Plot the notch‑filtered voltage trace.
        """
        self.channel0_Plot.clear()
        self.channel0_Plot.setLabel("left", "Voltage (mV)")
        self.channel0_Plot.setTitle(f"Raw + {mains} Hz notch")
        self.channel0_Plot.showGrid(x=True , y=True)
        self.channel0_Plot.plot(x, y, pen='Black')
        
    # def reset_plot(self):
    #     self.Plot.clear()
    #     self.raw_plot.clear()
    #     self.raw_plot.setLabel("left", "Voltage (mV)")
    #     self.raw_plot.setTitle(f"Raw + {mains} Hz notch")
    #     self.raw_plot.plot(x, y, pen='k')

    def _on_region_update(self, region, seg_idx, pw):
        if self.session is None:           # fallback → keep old behaviour
            return

        t0, t1 = sorted(region.getRegion())
        r_new  = self.session.refit_slice(seg_idx, t0=t0, t1=t1)
        self.current_results[seg_idx] = r_new
        self._redraw_segment(pw, r_new, region)   # << helper, unchanged
        self.fitUpdated.emit(seg_idx + 1, r_new)

    def _on_threshold_update(self, line):
        if self.session is None:
            return

        plot    = line._plot_widget
        seg_idx = line._seg_idx
        k90     = float(next(i for i in plot.items()
                            if getattr(i, "_tag","")=="k90_line").value())
        k10     = float(next(i for i in plot.items()
                            if getattr(i, "_tag","")=="k10_line").value())
        if k90 < k10:                       # guarantee k90 is higher
            k90, k10 = k10, k90

        region = next(itm for itm in plot.items()
                    if isinstance(itm, pg.LinearRegionItem))
        r_new = self.session.refit_slice(seg_idx, k90=k90, k10=k10)
        self.current_results[seg_idx] = r_new
        self._redraw_segment(plot, r_new, region)
        self.fitUpdated.emit(seg_idx + 1, r_new)

    def _redraw_segment(self, pw, r, region=None):
        """
        Draw (or re-draw) ONE clearance segment in *pw* using the
        up-to-date result-dict *r*.

        Parameters
        ----------
        pw : PlotWidget       – the tab we need to refresh
        r  : dict             – current segment dict (tau, seg_x …)
        region : LinearRegionItem or None
            If supplied, its limits are moved to match the new slice
            *without* emitting a signal (avoids recursion).
        """
        pw.clear()
        pw.addLegend(offset=(-100, 10))
        
        pk_time   = r["t0"]
        pre_start = pk_time - 1.0
        mask_pre  = (self.session.x >= pre_start) & (self.session.x <= pk_time)
        pre_x = self.session.x[mask_pre] - pk_time
        # pre_y = self.session.conc[mask_pre]
        raw_y = (self.session.filtered if
                 self.session.fit_domain == "voltage" else
                 self.session.conc)
        pre_y = raw_y[mask_pre]
        pw.plot(pre_x, pre_y,
                pen=pg.mkPen(width=1, color=(150,150,150)),
                name="pre-peak")
        
        # context window (grey)
        pw.plot(r["t_win"], r["y_win"],
                pen=pg.mkPen(width=1, color=(120,120,120)), name="context")

        # 2 · slice + fit (white / red)
        offset = r["slice_dt"]              # instead of slice_t0
        pw.plot(r["seg_x"] + offset, r["seg_y"], pen='k', name="slice")._tag = "raw"
        n_pts = r["n_data"]
        pw.plot(
            r["seg_x"][:n_pts] + offset,
            r["fitted"][:n_pts],
            pen='r', name="fit"
        )._tag = "fit"

        # 3 · k90 & k10 guide lines
        for y_val, (col, tag) in (
                (r["k90"], ((0, 0, 160), "k90_line")),
                (r["k10"], ('c', "k10_line"))
        ):
            line = pg.InfiniteLine(
                pos=y_val, angle=0, movable=True,
                pen=pg.mkPen(color=col, style=Qt.DashLine)
            )
            line._tag         = tag
            line._seg_idx     = r["peak"] - 1
            line._plot_widget = pw
            pw.addItem(line)
            line.sigPositionChangeFinished.connect(
                lambda ln=line: self._on_threshold_update(ln)
            )

        #  τ / R² label
        tau_str, lam_str = self._tau_lambda_strings(r)
        tw = r.get('tau_weighted')
        tw_str = f"{tw:.2f}" if tw is not None else "-"
        text = (
            f"τ={tau_str}s\n"
            f"R²={r['r2']:.3f}\n"
            f"λ={lam_str}s⁻¹\n"
            f"τ𝚠={tw_str}s"           # amplitude-weighted τ
        )

        lbl = pg.TextItem(
            text,
            color=(220,0,0), anchor=(0, 1)
        )

        
        lbl._tag = "lbl"
        pw.addItem(lbl)
        lbl.setPos(r["seg_x"][0] + offset, r["seg_y"][0] * 1.05)
        self._plot_comparison_on_axis(pw, r["peak"] - 1) 

        # 5 · move LinearRegionItem if one was passed in
        if region is not None:
            pw.addItem(region)
            try:
                region.blockSignals(True)
                region.setRegion((offset, offset + r["seg_x"][-1]))
            finally:
                region.blockSignals(False)
    
    def reset_image_items(self):
        layout = self.layout()
        # Remove old PlotWidget completely
        layout.removeWidget(self.Plot)
        self.Plot.deleteLater()

        # Recreate a clean PlotWidget
        self.Plot = pg.PlotWidget()
        layout.addWidget(self.Plot)

    # Plots the Overlay Traces
    def overlay_trace(self, x, y, peaks=None, *, t0=None,
                    pen=None, symbol_pen=None, symbol_brush=None):
        """
        Plot a *second* sweep on the Raw-ABF plot (channel0_Plot) so users
        can compare two recordings.  Uses a dashed magenta line by default
        and (optional) peak markers.

        Parameters
        ----------
        x, y : ndarray
            Time-axis and sweep values of the comparison trace (already in
            K⁺ units by the calling code).
        peaks : ndarray | None
            Sample indices of detected peaks in *y* (optional).
        pen : pg.mkPen | None
            Override the line style if you want something other than the
            default dashed-magenta.
        """

        if t0 is not None:
            x = x - t0

        from pyqtgraph import mkPen                              
        from PyQt5.QtCore import Qt

        # ---------- line (dashed magenta) ----------
        if pen is None:
            pen = mkPen((0, 0, 255), width=1, style=Qt.DashLine)
        self.channel0_Plot.plot(x, y, pen=pen, name="comparison")

        # ---------- optional peak markers ----------
        if peaks is not None and len(peaks):
            if symbol_pen is None:
                symbol_pen = None
            if symbol_brush is None:
                symbol_brush = (255, 0, 255)  # magenta triangles
            self.channel0_Plot.plot(
                x[peaks], y[peaks],
                pen=symbol_pen,
                symbol='t',
                symbolSize=6,
                symbolBrush=symbol_brush
            )

    def overlay_segments(
            self, other_results, other_session=None,
            pen_slice=None, pen_fit=None, pen_context=None, 
            align_mode="both", show_full_context=True
        ):
        """
        Overlay comparison segments with alignment options and full context.
        
        Args:
            align_mode: "baseline", "peak", or "both" (default)
            show_full_context: If True, shows peak and post-decay period
        """
        from pyqtgraph import mkPen
        from PyQt5.QtCore import Qt
        
        if pen_slice is None:
            pen_slice = mkPen((0, 0, 200), width=1, style=Qt.DashLine) #Trace (90/10% fit)
        if pen_fit is None:
            pen_fit = mkPen((200, 0, 200), width=1) # Curve Fit
        if pen_context is None:
            pen_context = mkPen((0, 0, 200), width=1, style=Qt.DotLine) #Extended before/after fit

        self.comparison_results = other_results
        self.comparison_session = other_session
        self.show_full_context = show_full_context

        n = min(len(self.current_results), len(other_results))
        
        for i in range(n):
            pw = self.tabs.widget(i + 1)
            r1 = self.current_results[i]
            r2 = other_results[i]

            # Calculate alignment transformations
            if align_mode == "baseline":
                baseline_key = "v_baseline" if self.session.fit_domain == "voltage" else "k_baseline"
                y_shift = r2[baseline_key] - r1[baseline_key]
                scale_factor = 1.0
                
            elif align_mode == "peak":
                peak_key = "v_peak" if self.session.fit_domain == "voltage" else "k_peak"
                y_shift = r2[peak_key] - r1[peak_key]
                scale_factor = 1.0
                
            elif align_mode == "both":
                if self.session.fit_domain == "voltage":
                    baseline1, peak1 = r1["v_baseline"], r1["v_peak"]
                    baseline2, peak2 = r2["v_baseline"], r2["v_peak"]
                else:
                    baseline1, peak1 = r1["k_baseline"], r1["k_peak"]
                    baseline2, peak2 = r2["k_baseline"], r2["k_peak"]
                
                amp1 = peak1 - baseline1
                amp2 = peak2 - baseline2
                
                scale_factor = amp1 / amp2 if amp2 != 0 else 1.0
                y_shift = baseline1 - (baseline2 * scale_factor)
            
            # Store transformation parameters
            r2["_y_shift"] = y_shift
            r2["_scale_factor"] = scale_factor
            r2["_align_mode"] = align_mode
            
            # Plot the comparison
            self._plot_comparison_full(pw, r1, r2, i, 
                                    pen_slice, pen_fit, pen_context,
                                    y_shift, scale_factor, show_full_context)


    def _plot_comparison_full(self, pw, r1, r2, seg_idx,
                            pen_slice, pen_fit, pen_context,
                            y_shift, scale_factor, show_full_context):
        """Plot the full comparison including peak and post-decay context."""
        
        if show_full_context and self.comparison_session:
            # Get the full context window (including peak and after decay)
            pk_idx = self.comparison_session.peaks[seg_idx]
            
            # Start from 2 seconds before peak
            pre_peak_sec = 2.0
            post_decay_sec = 5.0  # Show 5 seconds after the decay ends
            
            start_idx = max(0, pk_idx - int(pre_peak_sec * self.comparison_session.fs))
            
            # Find where the fitted decay ends and add extra time
            decay_end_time = r2["slice_dt"] + r2["seg_x"][-1]  # End of decay segment
            post_end_idx = pk_idx + int((decay_end_time + post_decay_sec) * self.comparison_session.fs)
            post_end_idx = min(post_end_idx, len(self.comparison_session.x) - 1)
            
            # Extract the full window
            if self.comparison_session.fit_domain == "voltage":
                full_y = self.comparison_session.filtered[start_idx:post_end_idx]
            else:
                full_y = self.comparison_session.conc[start_idx:post_end_idx]
                
            full_x = self.comparison_session.x[start_idx:post_end_idx] - self.comparison_session.x[pk_idx]
            
            # Apply transformations
            full_y_transformed = full_y * scale_factor + y_shift
            
            # Plot the full context (lighter/dotted line)
            pw.plot(full_x, full_y_transformed, pen=pen_context, name="comp context")
            
            # Mark the peak location
            peak_y_transformed = r2["y_win"][0] * scale_factor + y_shift  # Peak value
            pw.plot([0], [peak_y_transformed], 
                    pen=None, symbol='t', symbolBrush=(200, 0, 200),  # Changed 'v' to 't'
                    symbolSize=8, name="comp peak")
        
        # Plot the decay segment and fit (as before)
        offset = r2["slice_dt"]
        slice_x = r2["seg_x"] + offset
        n_pts = r2["n_data"]
        
        seg_y_corr = r2["seg_y"] * scale_factor + y_shift
        fitted_corr = r2["fitted"] * scale_factor + y_shift
        
        pw.plot(slice_x, seg_y_corr, pen=pen_slice, name="comp slice")
        pw.plot(slice_x[:n_pts], fitted_corr[:n_pts], pen=pen_fit, name="comp fit")


    def _plot_comparison_on_axis(self, pw, seg_idx, pen_slice=None, pen_fit=None):
        if self.comparison_results is None or seg_idx >= len(self.comparison_results):
            return
            
        from pyqtgraph import mkPen
        from PyQt5.QtCore import Qt

        r1 = self.current_results[seg_idx]
        r2 = self.comparison_results[seg_idx]

        if pen_slice is None:
            pen_slice = mkPen((0, 0, 200), width=1, style=Qt.DashLine)
        if pen_fit is None:
            pen_fit = mkPen((200, 0, 200), width=1)
        pen_context = mkPen((0, 0, 200), width=1, style=Qt.DotLine)

        # Get stored transformation parameters
        y_shift = r2.get("_y_shift", 0)
        scale_factor = r2.get("_scale_factor", 1.0)
        show_full_context = getattr(self, 'show_full_context', True)
        
        # Use the full plotting function
        self._plot_comparison_full(pw, r1, r2, seg_idx,
                                pen_slice, pen_fit, pen_context,
                                y_shift, scale_factor, show_full_context)



    def _on_overlay_threshold_update(self, line):
        if self.comparison_session is None:
            return
        plot    = line._plot_widget
        seg_idx = line._seg_idx

        k90_line = next(i for i in plot.items() if getattr(i, "_tag", "") == "comp_k90_line")
        k10_line = next(i for i in plot.items() if getattr(i, "_tag", "") == "comp_k10_line")

        shift = self.comparison_results[seg_idx].get("_y_shift", 0.0)

        k90 = float(k90_line.value()) - shift
        k10 = float(k10_line.value()) - shift
        if k90 < k10:
            k90, k10 = k10, k90

        r_new = self.comparison_session.refit_slice(seg_idx, k90=k90, k10=k10)
        self.comparison_results[seg_idx] = r_new
        self._redraw_complete_segment(plot, seg_idx)
        self.fitUpdated.emit(seg_idx + 1, r_new)


    def _on_overlay_region_update(self, region, seg_idx, pw):
        """Handle region dragging for the overlay trace"""
        if self.comparison_session is None:
            return
        
        t0, t1 = sorted(region.getRegion())
        r_new = self.comparison_session.refit_slice(seg_idx, t0=t0, t1=t1)
        self.comparison_results[seg_idx] = r_new
        
        # Redraw both traces
        self._redraw_complete_segment(pw, seg_idx)
        self.fitUpdated.emit(seg_idx + 1, r_new)

    def _redraw_complete_segment(self, pw, seg_idx):
        """Redraw both primary and overlay traces for a segment"""
        pw.clear()
        
        # Draw primary trace
        r1 = self.current_results[seg_idx]
        self._redraw_segment(pw, r1)
        
        # Draw overlay if exists
        if self.comparison_results and seg_idx < len(self.comparison_results):
            r2 = self.comparison_results[seg_idx]
            self._redraw_segment(pw, r2, is_overlay=True)
    
    def _baseline_key(self):
        return "v_baseline" if self.session.fit_domain == "voltage" else "k_baseline"


    def set_file(self, path: str | None):
        """Remember the current ABF’s basename and refresh titles."""
        self.file_name = os.path.basename(path) if path else None

        # update the raw-trace title immediately
        self.channel0_Plot.setTitle(
            f"{self.file_name}  –  Channel 0"
            if self.file_name else "Raw ABF – Channel 0"
        )


