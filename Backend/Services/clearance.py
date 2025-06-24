import pyabf
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit, minimize

# mV / decade for Nernst conversion. For every 10x increase in [K+],
# the voltage increases by this many mV.
DEFAULT_SLOPE = 58.0

#Normal brain tissue has 2.5mM K+ at rest. This is our zero point.
RESTING_K_MM  = 2.5           

# Default length of the slice taken after each peak for analysing the decay
DECAY_WINDOW_S = 15    

# We'll remove electrical noise at 50 Hz 
NOTCH_WIDTH = 1             

# How long before the peak to look for a baseline
BASELINE_WINDOW_PRE_SEC = 3.0   

# longest slice sent to the exp‑fit
MAX_FIT_AFTER_90_SEC = 15.0  

 # analyse at most the first 4 peaks
MAX_PEAKS = 4    

# reserved for shifting the fit window later if needed
FIT_START_OFFSET_SEC = 0

class ClearanceSession:
    def __init__(self, file_path, # ABF file path
                 sweep: int = 0, # sweep index to load
                 slope: float = DEFAULT_SLOPE, # how sensitive the electrode is
                 resting_k: float = RESTING_K_MM, # resting [K+] in mM
                 decay_window_sec: float = DECAY_WINDOW_S, # how long to analyse after each peak
                 fit_method: str = "cheb",
                 n_exp: int = 2,
                 fit_domain: str = "voltage"): 
        
        self.fit_method = fit_method.lower()
        self.n_exp = max(1, int(n_exp))
        self.fit_domain = fit_domain.lower()
        
        # Load ABF and select first sweep
        self.abf = pyabf.ABF(file_path)
        print(f"[DEBUG] {file_path} contains {self.abf.sweepCount} sweep(s)")
        self.sweep_count = self.abf.sweepCount

        if not (0 <= sweep < self.sweep_count):
            raise ValueError(f"Invalid sweep index {sweep} (0..{self.sweep_count-1})")
        self._set_sweep(sweep)

        # Raw voltage trace & time
        self.y_raw = self.abf.sweepY.copy() #Voltage measurements in mV
        self.x     = self.abf.sweepX # Time in seconds
        self.fs    = self.abf.sampleRate # Sampling frequency in Hz (How many samples per second)
        print(f"{file_path} — sweeps={self.abf.sweepCount} sampleRate={self.fs} Hz")

        # Converts the 15 s constant into samples so later code can slice arrays
        self.decay_window_samples = int(decay_window_sec * self.fs)

        # Nernst‐fit parameters
        self.slope      = slope
        self.resting_k  = resting_k

        # Estimate intercept from the first 0.5 s baseline
        baseline_mV   = self._robust_baseline()
        self.intercept = baseline_mV - slope * np.log10(resting_k)
        print(f"baseline mV={baseline_mV:.2f} intercept={self.intercept:.2f}")

        # Placeholders
        self.filtered = None   # filtered voltage (mV)
        self.conc     = None   # converted [K+] trace (mM)
        self.peaks    = None   # indices of detected peak
    
    def _set_sweep(self, idx: int):
        """(Re)load data arrays for the selected sweep."""
        self.abf.setSweep(idx)
        self.sweep_index = idx                     
        self.y_raw = self.abf.sweepY.copy()
        self.x     = self.abf.sweepX
        self.fs    = self.abf.sampleRate
    
    def change_sweep(self, idx: int):
        self._set_sweep(idx)
        # downstream caches must be cleared
        self.filtered = None
        self.conc     = None
        self.peaks    = None

        
    def _robust_baseline(self, search_sec=5.0, window_sec=0.5) -> float:
        """
        Scans the first 5 seconds of the sweep in 0.5 s windows with 50 % overlap,
        takes the mean of each window and returns the lowest mean.
        """
        win  = int(window_sec * self.fs)
        stop = int(search_sec  * self.fs)
        step = win // 2                        # 50 % overlap
        means = [
            self.y_raw[i:i + win].mean()
            for i in range(0, max(1, stop - win), step)
        ]
        idx = np.argmin(means) * step
        return self.y_raw[idx:idx + win].mean()
    
    
    @staticmethod
    def _multi_exp_decay(t, *params):
        """
        params = [A1, tau1, A2, tau2, ..., C]
        """
        *pairs, C = params
        y = np.zeros_like(t, dtype=float)
        for A, tau in zip(pairs[::2], pairs[1::2]):
            y += A * np.exp(-t / tau)
        return y + C

    def _chebyshev_fit_multi(self, t, y, n=2, p0=None, C_lower=-np.inf, maxiter=6000):
        if p0 is None:                      # crude first guess
            A_guess = (y[0] - y[-1]) / n
            taus    = np.geomspace(0.1, 1.0, n)  # s
            p0 = [A_guess if i%2==0 else taus[i//2] for i in range(2*n)] + [y[-1]]
        def objective(p):
            err = self._multi_exp_decay(t, *p) - y
            return np.sqrt(np.mean(err**2))
        cons = [{'type': 'ineq', 'fun': lambda p, j=j: p[2*j+1] - 1e-6}   # τ_j ≥ 0
            for j in range(n)] + [
            {'type': 'ineq', 'fun': lambda p: C_lower - p[-1]}]        # C ≤ y10

        res = minimize(objective, p0, method="SLSQP",
                    constraints=cons, options=dict(maxiter=maxiter))
        if not res.success:
            raise RuntimeError(res.message)
        return res.x
 

    @staticmethod
    def _exp_decay(t, A, tau, C):
        """Return A·exp(-t/τ)+C"""
        return A * np.exp(-t / tau) + C

    def apply_notch(self, freq: float = 50.0, width: float = NOTCH_WIDTH) -> np.ndarray:
        """
        Apply a ±width Hz notch around freq Hz to the raw voltage,
        convert to absolute [K+] (mM) for analysis, and return
        a zero-baselined copy for plotting.
        """
        # 1) Notch‐filter the raw voltage
        nyq = self.fs / 2 #Nyquist frequency (half the sampling rate)
        low = (freq - width) / nyq 
        high = (freq + width) / nyq
        b, a = butter(2, [low, high], btype='bandstop') 
        self.filtered = filtfilt(b, a, self.y_raw)

        # 2) Convert to absolute [K+] via inverted Nernst
        # - Takes voltage readings in mV
        # - Subtracts the intercept
        # - Divides by the slope
        # - Raises 10 to the power of the result
        # Results in [K+] in mM.
        conc_abs = 10 ** ((self.filtered - self.intercept) / self.slope)
        self.conc = conc_abs

        # 3) Compute resting K+ once, subtract for plotting
        rest_mask = self.x < 0.5 # first 0.5 s of the sweep
        k_rest = np.mean(conc_abs[rest_mask]) #Calculate the mean [K+] in the first 0.5 s
        self.conc_zeroed = conc_abs - k_rest #only for plotting

        print(f"Resting [K+] = {k_rest:.2f} mM; plotting trace with rest set to 0.")

        # 4) Return the zeroed trace for GUI plotting
        return self.conc_zeroed


    def detect_peaks(        self,
        prominence: float | None = None,
        distance  : float | None = None,
        auto_flip : bool = True) -> np.ndarray:
        """
        Detect peaks in the converted [K+] trace.
        prominence in mM, distance defaults to 2 s.
        Prominence: How much the peak must stand out (default: 20% of signal range)
        Distance: Minimum time between peaks (default: 2 seconds)
        """
        if self.conc is None:
            raise RuntimeError("Call apply_notch() before detect_peaks().")

        if distance is None:
            distance = int(2 * self.fs)

        if prominence is None:
            rng = float(self.conc.max() - self.conc.min())
            prominence = max(0.2 * rng, 0.1) 

        def _run_peak_finder(sig, label):
            peaks, _ = find_peaks(
                sig,
                prominence=prominence,
                distance=distance
            )
            print(  
                f"→ {label} peaks: {len(peaks)} "
                f"with prominence ≥{prominence:.2f} mM "
                f"at {[round(p/self.fs,2) for p in peaks]} s"
            )            
            return peaks
        
        self.peaks = _run_peak_finder(self.conc, "upward")

        if auto_flip and len(self.peaks) == 0:
            peaks_down = _run_peak_finder(-self.conc, "downward")
            if len(peaks_down):
                self.conc *= -1
                self.peaks = peaks_down
                print("• Trace auto flipped (downward peaks chosen)")

        if len(self.peaks) == 0:
            print("• No peaks detected — adjust prominence or inspect trace.")

        self.peaks = self.peaks[:MAX_PEAKS]

        return self.peaks
    
    @staticmethod
    def _first_cross(t, y, y_thr):
        """linear-interpolated time where y(t) falls below y_thr"""
        idx = np.flatnonzero(y <= y_thr)[0]
        if idx == 0:
            return 0.0, idx
        # interpolate between idx-1 and idx
        t0, t1 = t[idx-1], t[idx]
        y0, y1 = y[idx-1], y[idx]
        frac = (y_thr - y0)/(y1 - y0)
        return t0 + frac*(t1 - t0), idx

    def clearance_test(self) -> list[dict]:
        """
        For every peak previously detected in self.peaks: 
        - Cut out just the 90 % and 10 % decay windows
        - Fit an exponential decay to the data
        - Collect t (time-constant) and λ=1/τ (clearance rate)

        Returns:
        - List of dicts. One dict per peak.
        """

        # Check if the peaks were detected
        if self.peaks is None:
            raise RuntimeError("Call detect_peaks() before clearance_test().")

        results = [] # final list of dicts
        fit_trace = self.filtered if self.fit_domain == "voltage" else self.conc

        # Loop over the detected peaks
        for i, pk in enumerate(self.peaks, start=1):
            # —──────────────────────── 1 · window & thresholds ─────────────────────────
            # 30 second window after the peak (y_win, t_win) to search for 90% and 10%
            # y_win - [K+] axis (mM) relative to the peak
            # t_win - 0...30 s (time axis relative to the peak)

            # Measure the baseline voltage before the peak (median of a 3 s window)
            # Median is used because it is resistant to noise spikes
            start_base = max(0, pk - int(BASELINE_WINDOW_PRE_SEC * self.fs))
            v_baseline = float(np.median(self.filtered[start_base:pk]))
            v_peak     = float(self.filtered[pk]) #voltage at the peak
            delta_v    = v_peak - v_baseline
            print(f"peak #{i}: V_base={v_baseline:.2f} mV, V_peak={v_peak:.2f} mV, ΔV={delta_v:.2f} mV")

            # convert those voltages to [K⁺] (mM)
            # log10([K⁺] = (V - intercept) / slope
            k_baseline = 10 ** ((v_baseline - self.intercept) / self.slope)
            k_peak     = 10 ** ((v_peak     - self.intercept) / self.slope)

            # Compute Δ[K⁺] so we can derive the 90 % and 10 % thresholds
            delta_k = k_peak - k_baseline

            if self.fit_domain == "voltage":
                # work in VOLTAGE (mV) all the way down  ↓↓↓
                y_src   = self.filtered
                y_peak  = v_peak
                y_base  = v_baseline
                delta_y = delta_v
                y90 = y_base + 0.9 * delta_y
                y10 = y_base + 0.1 * delta_y
                k90_to_store = y90
                k10_to_store = y10
            else:                              # concentration
                y_src   = self.conc
                y_peak  = k_peak
                y_base  = k_baseline
                delta_y = delta_k
                y90 = k90                      # already computed
                y10 = k10
                k90_to_store = k90   
                k10_to_store = k10

            end_search = min(pk + int(30 * self.fs), len(y_src))
            y_win = y_src[pk:end_search]
            t_win = self.x[pk:end_search] - self.x[pk]

            # Create k90 and k10 thresholds - always make k90 > k10
            if delta_k >= 0:
                k90 = k_baseline + 0.9 * delta_k 
                k10 = k_baseline + 0.1 * delta_k
            else:
                k90 = k_baseline + 0.1 * delta_k
                k10 = k_baseline + 0.9 * delta_k

            # Skip weird 'peaks' that point the wrong way
            if delta_k < 0:
                print(f"peak #{i}: negative Δ[K⁺] ({delta_k:.3f} mM) – segment skipped")
                continue          

            # Locate where the trace crosses the thresholds
            # np.flatnonzero gives sample indices where the condition is true
            cross90 = np.flatnonzero(y_win <= y90)
            cross10 = np.flatnonzero(y_win <= y10)
            if len(cross90) == 0 or len(cross10) == 0:
                # peak never decayed to ≤ 10 % – skip
                continue

            t90, idx90 = self._first_cross(t_win, y_win, y90)
            t10, idx10 = self._first_cross(t_win, y_win, y10)
            decay_time = t10 - t90
            y10 = y_src[pk + idx10]

            # Slice the 90/10 % region out of the whole sweep
            start_fit = max(pk + idx90 - 1, pk)
            end_fit   = pk + idx10
            seg_x = self.x[start_fit:end_fit + 1] - self.x[start_fit]
            seg_y = y_src[start_fit:end_fit + 1]
            upper_C = y10
            

            if self.n_exp == 1:
                p0       = [seg_y[0] - seg_y[-1], 1.0, seg_y[-1]]
                lower_b  = [0, 1e-6, -np.inf]
                upper_b  = [np.inf, np.inf, y10]       # C must stay ≤ 10 % level
                fit_fun  = self._exp_decay
            else:
                amp_guess = (seg_y[0] - seg_y[-1]) / self.n_exp
                taus      = np.geomspace(0.2, 1.0, self.n_exp)   # seconds
                parts     = np.ravel(np.column_stack([np.full(self.n_exp, amp_guess), taus]))
                p0        = parts.tolist() + [seg_y[-1]]
                lower_b   = [0, 1e-6] * self.n_exp + [-np.inf]
                upper_b   = [np.inf, np.inf] * self.n_exp + [y10]
                fit_fun   = self._multi_exp_decay

            if self.fit_method == "cheb":
                if self.n_exp == 1:
                    p_fit = self._chebyshev_fit(seg_x, seg_y,
                                                p0=p0, k10_bound=y10)
                else:
                    p_fit = self._chebyshev_fit_multi(seg_x, seg_y,
                                                    n=self.n_exp,
                                                    p0=p0, C_lower=y10)
                    
                if self.n_exp == 1:
                    A_fit, tau_fit, C_fit = p_fit
                    amps = [A_fit]
                    taus = [tau_fit]
                else:
                    *coeffs, C_fit = p_fit
                    amps = coeffs[::2]
                    taus = coeffs[1::2]
                    tau_fit = None  
                            

            amps, taus = zip(*sorted(zip(amps, taus), key=lambda p: p[1], reverse=True))
            amps = list(amps); taus = list(taus)
            lam = [1.0 / t for t in taus] 

            T3 = None
            if len(taus) >= 2:
                T1, T2 = taus[:2]
                T3 = (T1 * T2) / (T1 + T2)

            # ─── 4 · make the fitted curve & optionally extend it ──────────────────
            fit_x = seg_x
            fit_y = fit_fun(fit_x, *p_fit)

            # ––– compute when the MODEL reaches 10 % of the initial Δ ---
            def t_to_level(A_list, tau_list, C, target):
                """
                Solve Σ A_i e^(-t/τ_i) + C = target  (simple Newton–Raphson, fine for n=2)
                Returns None if the curve never gets that low.
                """
                from math import exp
                t = 0.0
                for _ in range(80):           # max 80 iterations
                    f  = sum(A*exp(-t/tau) for A, tau in zip(A_list, tau_list)) + C - target
                    if abs(f) < 1e-4:
                        return t
                    df = sum(-A/tau*exp(-t/tau) for A, tau in zip(A_list, tau_list))
                    if df == 0:
                        break
                    t -= f/df
                return None

            t10_model = t_to_level(amps, taus, C_fit, y10)
            if t10_model and t10_model > fit_x[-1] and t10_model <= MAX_FIT_AFTER_90_SEC:
                extra_t = np.linspace(fit_x[-1] + 1 / self.fs, t10_model,
                                    int((t10_model - fit_x[-1]) * self.fs))
                fit_x   = np.concatenate([fit_x, extra_t])
                fit_y   = np.concatenate([fit_y, fit_fun(extra_t, *p_fit)])
                decay_time = t10_model
            else:
                decay_time = seg_x[-1]


            # assess the goodness of fit (R2)
            n_data = len(seg_y)
            ss_res = np.sum((seg_y - fit_y[:n_data]) ** 2) #residual error
            ss_tot = np.sum((seg_y - seg_y.mean()) ** 2) #total variance
            r2     = 1.0 - ss_res / ss_tot # 1 - SSR/SST


            # package everything into a dict
            results.append({
                "peak":        i,
                "t0"      : float(self.x[pk]), # time of the peak
                # voltages & concentrations
                "v_baseline":  v_baseline,
                "v_peak":      v_peak,
                "delta_v":     delta_v,
                "k_baseline":  k_baseline,
                "k_peak":      k_peak,
                "delta_k":     delta_k,
                # fitted parameters
                "tau":         taus,
                "lambda":      lam,
                "T3"    : T3, 
                "r2":          float(r2),
                "amps":       amps,
                "C":          C_fit,   
                # timing
                "decay_time":  float(decay_time),
                # raw + fit arrays for plotting
                "slice_x":     seg_x,
                "slice_y":     seg_y,
                "fit_x":       fit_x,
                "fit_y":       fit_y,
                # context for GUI re-editing
                "k90":         float(k90_to_store),
                "k10":         float(k10_to_store),
                "full_x":      self.x[pk:end_search] - self.x[pk],
                "full_y":      y_src[pk:end_search],
                "t_win":       t_win,
                "y_win":       y_win,
                "n_data": n_data,
                "slice_t0": float(self.x[start_fit]), 
                "slice_dt"  : float((start_fit - pk) / self.fs),
                "seg_x":       seg_x,
                "seg_y":       seg_y,
                "fitted":      fit_y,
            })
            
        self.results = results
        return results

    def full_signal(self):
        """
        Returns (x, k_conc, peaks, t0) where t0 is the time of the
        first detected peak, so you can align traces.
        """
        if self.conc is None:
            raise RuntimeError("Call apply_notch() first.")
        if self.peaks is None:
            self.detect_peaks()
        t0 = float(self.x[self.peaks[0]])
        return self.x, self.conc, self.peaks, t0
    
    def _chebyshev_fit(self, t, y, p0, k10_bound, maxfev=6000):
            import warnings
            from scipy.optimize import minimize
            print("[DEBUG] Chebyshev fitter called")

            if p0 is None:
                p0 = np.array([y[0] - y[-1], 1.0, y[-1]])

            def objective(p):
                return np.sqrt(np.mean((self._multi_exp_decay(t, *p) - y)**2))
        
            constraints = ({'type': 'ineq', 'fun': lambda p: k10_bound - p[2]})

            res = minimize(objective, p0, method="SLSQP", constraints=constraints, options=dict(maxiter=maxfev))

            if not res.success:
                raise RuntimeError(res.message)
            return res.x

    def refit_slice(
            self,
            seg_idx: int,
            t0 : float | None = None,
            t1 : float | None = None,
            k90: float | None = None,
            k10: float | None = None
        ) -> dict:
        """
        Re-compute one clearance segment after the GUI was edited.

        • t0 / t1 come from dragging the vertical LinearRegionItem
        • k90 / k10 come from dragging the horizontal 90 % / 10 % lines

        Returns a dict in the *exact* format produced by clearance_test().
        Safe to run in the GUI thread – pure NumPy.
        """
        r_old = self.results[seg_idx]

        # ── 1 · full arrays (relative to peak, t = 0) ─────────────────────────
        t_full = r_old["full_x"]
        y_full = r_old["full_y"]

        # ── 2 · mask the new slice ───────────────────────────────────────────
        if k90 is None or k10 is None:                 # region-drag
            mask = (t_full >= t0) & (t_full <= t1)
        else:                                          # threshold-drag
            idx90 = np.flatnonzero(y_full <= k90)[0]
            idx10 = np.flatnonzero(y_full <= k10)[0]
            mask  = np.zeros_like(t_full, dtype=bool)
            mask[idx90:idx10 + 1] = True

        x_slice = t_full[mask]
        y_slice = y_full[mask]
        x_local = x_slice - x_slice[0]                 # start at 0 s

        # ── 3 · build initial guess & model (identical to clearance_test) ────
        if self.n_exp == 1:
            p0      = [y_slice[0] - y_slice[-1], 1.0, y_slice[-1]]
            fit_fun = self._exp_decay
        else:
            amp_g   = (y_slice[0] - y_slice[-1]) / self.n_exp
            taus_g  = np.geomspace(0.2, 1.0, self.n_exp)
            parts   = np.ravel(np.column_stack([np.full(self.n_exp, amp_g), taus_g]))
            p0      = parts.tolist() + [y_slice[-1]]
            fit_fun = self._multi_exp_decay

        k10_active = k10 if k10 is not None else r_old["k10"]

        # ── 4 · run the fitter of choice ─────────────────────────────────────
        if self.fit_method == "cheb":
            if self.n_exp == 1:
                p_fit = self._chebyshev_fit(
                    x_local, y_slice, p0=p0, k10_bound=k10_active)
            else:
                p_fit = self._chebyshev_fit_multi(
                    x_local, y_slice, n=self.n_exp,
                    p0=p0, C_lower=k10_active)
        else:                                          # SciPy LM (curve_fit)
            p_fit, _ = curve_fit(fit_fun, x_local, y_slice, p0=p0, maxfev=6000)

        # ── 5 · unpack parameters ────────────────────────────────────────────
        if self.n_exp == 1:
            A_fit, tau_fit, C_fit = p_fit
            amps  = [A_fit]
            taus  = [tau_fit]
        else:
            *coeffs, C_fit = p_fit
            amps = coeffs[::2]
            taus = coeffs[1::2]

        y_fit = fit_fun(x_local, *p_fit)

        # Goodness-of-fit
        ss_res = np.sum((y_slice - y_fit) ** 2)
        ss_tot = np.sum((y_slice - y_slice.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot

        # ── 6 · package updated result dict ──────────────────────────────────
        r_new = r_old.copy()
        r_new.update({
            "seg_x" : x_local,
            "seg_y" : y_slice,
            "fitted": y_fit,
            "tau"   : taus,
            "lambda": [1.0 / t for t in taus],
            "amps"  : amps,
            "C"     : C_fit,
            "r2"    : float(r2),
            "n_data": len(y_slice),
            "decay_time": float(x_local[-1]),
            "slice_dt"  : float(x_slice[0]),
            "last_bounds": (x_slice[0], x_slice[-1]),
        })
        if k90 is not None and k10 is not None:
            r_new.update({"k90": k90, "k10": k10})

        # keep master list in sync & hand back
        self.results[seg_idx] = r_new
        return r_new


    def validate_experiment(self, pre_calibration: dict = None, 
                       post_calibration: dict = None) -> dict:
        """
        Perform comprehensive validation of the experiment.
        
        Args:
            pre_calibration: {'k_concentrations': [...], 'voltages': [...]}
            post_calibration: {'k_concentrations': [...], 'voltages': [...]}
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # 1. Check current slope
        slope_valid, slope_msg = self.validate_electrode_slope(self.slope)
        results['checks']['slope'] = {'valid': slope_valid, 'message': slope_msg}
        if not slope_valid:
            results['errors'].append(slope_msg)
            results['valid'] = False
        
        # 2. Check baseline stability
        baseline_valid, baseline_msg = self.validate_baseline_stability()
        results['checks']['baseline'] = {'valid': baseline_valid, 'message': baseline_msg}
        if not baseline_valid:
            results['warnings'].append(baseline_msg)
        
        # 3. Check calibration data if provided
        if pre_calibration:
            pre_valid, pre_slope, pre_int, pre_msg = self.validate_calibration_curve(
                pre_calibration['k_concentrations'], 
                pre_calibration['voltages']
            )
            results['checks']['pre_calibration'] = {
                'valid': pre_valid, 
                'slope': pre_slope,
                'message': pre_msg
            }
            if not pre_valid:
                results['errors'].append(f"Pre-calibration: {pre_msg}")
                results['valid'] = False
                
            # If post-calibration provided, check stability
            if post_calibration:
                post_valid, post_slope, post_int, post_msg = self.validate_calibration_curve(
                    post_calibration['k_concentrations'], 
                    post_calibration['voltages']
                )
                results['checks']['post_calibration'] = {
                    'valid': post_valid,
                    'slope': post_slope, 
                    'message': post_msg
                }
                
                if pre_valid and post_valid:
                    stability_valid, stability_msg = self.validate_calibration_stability(
                        pre_slope, post_slope
                    )
                    results['checks']['calibration_stability'] = {
                        'valid': stability_valid,
                        'message': stability_msg
                    }
                    if not stability_valid:
                        results['warnings'].append(stability_msg)
        
        # 4. Check if peaks were detected
        if self.peaks is None or len(self.peaks) == 0:
            results['warnings'].append("No peaks detected in the trace")
        
        # 5. Check fit quality for each peak
        if hasattr(self, 'results') and self.results:
            poor_fits = [r for r in self.results if r['r2'] < 0.95]
            if poor_fits:
                results['warnings'].append(
                    f"{len(poor_fits)} peak(s) with poor fit quality (R² < 0.95)"
                )
        
        return results

    def validate_electrode_slope(self, slope: float) -> tuple[bool, str]:
        """
        Validate that electrode slope is within acceptable range.
        
        Args:
            slope: Electrode slope in mV/decade
            
        Returns:
            (is_valid, message)
        """
        MIN_SLOPE = 53.0
        MAX_SLOPE = 60.0
        
        if MIN_SLOPE <= slope <= MAX_SLOPE:
            return True, f"Electrode slope {slope:.1f} mV/decade is acceptable"
        else:
            return False, f"Electrode slope {slope:.1f} mV/decade outside acceptable range (53-60 mV)"

    def validate_baseline_stability(self, baseline_duration_s: float = 5.0, 
                                max_drift_mV: float = 2.0,
                                max_noise_mV: float = 1.0) -> tuple[bool, str]:
        """
        Check if baseline is stable (low drift and noise).
        
        Args:
            baseline_duration_s: Duration to check (seconds)
            max_drift_mV: Maximum acceptable drift
            max_noise_mV: Maximum acceptable noise (std dev)
            
        Returns:
            (is_valid, message)
        """
        if self.filtered is None:
            self.apply_notch()
        
        # Get baseline segment
        baseline_samples = int(baseline_duration_s * self.fs)
        baseline = self.filtered[:baseline_samples]
        time = self.x[:baseline_samples]
        
        # Check drift (linear regression)
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(time, baseline)
        drift_mV = abs(slope * baseline_duration_s)  # total drift over period
        
        # Check noise
        detrended = baseline - (slope * time + intercept)
        noise_mV = np.std(detrended)
        
        issues = []
        if drift_mV > max_drift_mV:
            issues.append(f"drift {drift_mV:.2f} mV > {max_drift_mV} mV")
        if noise_mV > max_noise_mV:
            issues.append(f"noise {noise_mV:.2f} mV > {max_noise_mV} mV")
        
        if issues:
            return False, f"Baseline unstable: {', '.join(issues)}"
        else:
            return True, f"Baseline stable (drift: {drift_mV:.2f} mV, noise: {noise_mV:.2f} mV)"

    def validate_calibration_curve(self, k_concentrations: list[float], 
                                voltages: list[float]) -> tuple[bool, float, float, str]:
        """
        Validate calibration curve from known K+ concentrations.
        
        Args:
            k_concentrations: Known [K+] in mM (e.g., [2.5, 5, 10, 15, 30, 100])
            voltages: Measured voltages in mV
            
        Returns:
            (is_valid, slope, intercept, message)
        """
        from scipy import stats
        
        # Fit Nernst equation: V = intercept + slope * log10([K+])
        log_k = np.log10(k_concentrations)
        slope, intercept, r_value, _, _ = stats.linregress(log_k, voltages)
        r_squared = r_value ** 2
        
        # Check slope range
        slope_valid = 53 <= slope <= 60
        
        # Check linearity (R² should be very high for good electrode)
        linearity_valid = r_squared >= 0.98
        
        issues = []
        if not slope_valid:
            issues.append(f"slope {slope:.1f} mV outside 53-60 mV range")
        if not linearity_valid:
            issues.append(f"poor linearity (R²={r_squared:.3f})")
        
        is_valid = slope_valid and linearity_valid
        
        if is_valid:
            message = f"Calibration valid (slope: {slope:.1f} mV, R²: {r_squared:.3f})"
        else:
            message = f"Calibration invalid: {', '.join(issues)}"
        
        return is_valid, slope, intercept, message

    def validate_calibration_stability(self, pre_slope: float, post_slope: float) -> tuple[bool, str]:
        """
        Check if electrode calibration remained stable throughout experiment.
        
        Args:
            pre_slope: Slope before experiment (mV/decade)
            post_slope: Slope after experiment (mV/decade)
            
        Returns:
            (is_valid, message)
        """
        deviation = abs(post_slope - pre_slope) / pre_slope * 100
        
        if deviation <= 20:
            return True, f"Calibration stable (deviation: {deviation:.1f}%)"
        else:
            return False, f"Calibration unstable (deviation: {deviation:.1f}% > 20%)"
        

