import pyabf
import numpy as np
from scipy.signal import welch, spectrogram, iirnotch, filtfilt, butter
from scipy.ndimage import uniform_filter1d

class Extracellular:
    def __init__(self, filepath):
        self.abf = pyabf.ABF(filepath)
        self.signal = None
        self.time = None
        self.fs = None
        
    def bandpass_filter(self, data, fs, lowcut, highcut, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def highpass_filter(self, data, fs, cutoff=1.0, order=2):
        b, a = butter(order, cutoff / (fs / 2), btype='high')
        return filtfilt(b, a, data)

    def lowpass_filter(self, data, fs, cutoff=100.0, order=3):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
        
    def apply_notch(self, signal, fs, bandpassfilter):
        print("[DEBUG] inside apply_notch function")
        for freq in [50.0, 100.0]:
            if fs < 2 * freq:
                raise ValueError(f"Sampling rate too low for {freq} Hz notch filter.")
            b, a = iirnotch(w0=freq, Q=60, fs=fs)
            signal = filtfilt(b, a, signal)
        if bandpassfilter:
            signal = self.bandpass_filter(signal, fs, lowcut=1.0, highcut=100.0, order=3)
            return signal
        else:
            return signal

    def segment_raw_signal(self, baseline_data=None, highk_data=None, recovery_data=None, bandpassfilter= None, highpassfilter= None, lowpassfilter= None):
        print("[DEBUG] inside segment_raw_signal function")
        fs = self.abf.dataRate

        if baseline_data is not None and highk_data is not None and recovery_data is not None:
            print("[DEBUG] using provided segments")
            segments = {
                "baseline": baseline_data,
                "high_k": highk_data,
                "recovery": recovery_data
            }
        else:
            raise ValueError("Manual segmentation required. Provide baseline, high_k, and recovery data.")
           
        # Always apply notch filter
        print("[DEBUG] Applying filters as needed")
        print("[DEBUG] Segment contents:", segments)
        filtered_segments = {}
        for k, (t, y) in segments.items():
            y = self.apply_notch(np.asarray(y), fs, bandpassfilter)
            if lowpassfilter == True:
                y = self.lowpass_filter(np.asarray(y), fs)
            if highpassfilter == True:
                y = self.highpass_filter(np.asarray(y), fs)
                
            filtered_segments[k] = (t, y)
            
        Extracellular.stored_segments = filtered_segments
        return filtered_segments
    
    def Calc_globalMinMax(self, spectrograms):
        all_vals = []
        # Goes through each Spectogram
        for spec_data in spectrograms.values():
            # Extracts Data
            f = spec_data["f"]
            Sxx_db = spec_data["Sxx_db"]

            print(f"Checking segment with f.shape = {f.shape}, Sxx_db.shape = {Sxx_db.shape}")
            # Checks if the shape matches between the freq and Sxx_db - Skips that segment
            if f.shape[0] != Sxx_db.shape[0]:
                print(f"[ERROR] f shape: {f.shape}, Sxx_db shape: {Sxx_db.shape}")
                continue  # Skip this segment
            
            # Masks the standardisation of global max and min to a 0-100hz band
            freq_mask = (f >= 0) & (f <= 100)
            print(f"f min: {f.min()}, f max: {f.max()}, freq_mask sum: {np.sum(freq_mask)}")
            Sxx_db_limited = Sxx_db[freq_mask, :]
            
            # Flattens and collect valid values preparing it for placement into a 1D array
            if Sxx_db_limited.size > 0:
                all_vals.append(Sxx_db_limited.flatten())

        if not all_vals:
            raise ValueError("No valid spectrogram values in 0–100 Hz range.")
        
        # Places all values into a 1D array to find the global min and max
        all_vals = np.concatenate(all_vals)
        return np.min(all_vals), np.max(all_vals)
    
    def Extract_SpectogramData(self, segments, sample_rate=None):
        # Checks for inputted Data
        if segments is None:
            segments = Extracellular.get_stored_segments()
        if segments is None:
            raise ValueError("No stored segments available.")

        if sample_rate is None:
            sample_rate = self.abf.dataRate

        # Creates Results Arrays
        spectrogram_results = {}
        # Loops through time array (t_seg) and signal array(y_seg)
        for label, (t_seg, y_seg) in segments.items():
            if len(y_seg) < 256:
                continue

            # Determine appropriate nperseg based on signal length
            # For sample rate it gives the best range to make a smooth picture
            if sample_rate == 5000:    
                max_nperseg = 4096
                min_nperseg = 2048
            elif sample_rate == 10000:
                max_nperseg = 8192
                min_nperseg = 4096
            duration_sec = len(y_seg) / sample_rate
            # Clamp to a reasonable range
            target_bins = max(30, min(int(duration_sec * 40), 100))
            
            # Estimate a good nperseg for the current segment
            self.adaptive_nperseg = max(min_nperseg, min(max_nperseg, len(y_seg) // target_bins))
            
            print("[DEBUG] Nperseg: ", self.adaptive_nperseg)
            print("[DEBUG] Target_Bins Value:", target_bins)
                        
            f, t, Sxx = spectrogram(
                y_seg,
                fs=sample_rate,
                window='hamming',
                nperseg=self.adaptive_nperseg,
                nfft= 8192,
                noverlap=int(self.adaptive_nperseg * 0.75),
                mode='psd'
            )

            Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Avoid log(0)

            spectrogram_results[label] = {
                "t": t, 
                "f": f, 
                "Sxx_db": Sxx_db
            }
            
            print(spectrogram_results)

        return spectrogram_results
    
    def display_heatmap(self, spectograms=None, global_min=None, global_max=None):
        if spectograms is None:
            spectograms = self.Extract_SpectogramData()
            
        heatmap_results = {}
        for label, spec_data in spectograms.items():
            f = spec_data["f"]
            t = spec_data["t"]
            Sxx_db = spec_data["Sxx_db"]
            
            Spec_min = global_min
            Spec_max = global_max
            
            # Peak detection
            power = Sxx_db.max()
            idx = np.unravel_index(np.argmax(Sxx_db), Sxx_db.shape)
            peak_freq = f[idx[0]]
            peak_time = t[idx[1]]
            
            heatmap_results[label] = (f, t, Sxx_db, peak_freq, peak_time, power, Spec_min, Spec_max)
            
        return heatmap_results
    
    def display_spectrum(self, segments=None, fs=None, max_freq=100, nperseg= 4096):
        # Fetches segments if they don't get passed through properly
        if segments is None:
            segments = Extracellular.get_stored_segments()
            print("Fetched stored segments:", segments)
        # If nothing is fetched print error
        if not segments:
            raise ValueError("No segmented data available.")
        # Checks if there is a sampling rate
        if fs is None:
            if hasattr(self, "abf") and hasattr(self.abf, "dataRate"):
                fs = self.abf.dataRate
            else:
                raise ValueError("Sampling frequency 'fs' not provided and could not be inferred from self.abf.")

        # Creates array for results to append too
        results = {}
        scale_factor = 1
        # scale_factor = 1e-2  # To match reference plot unit scale (μV² × 10⁻²)

        for label, (t_seg, y_seg_list) in segments.items():
            if not isinstance(y_seg_list, list):
                y_seg_list = [y_seg_list]  # Wrap single trace in a list

            psds = []
            freqs = None

            for y_seg in y_seg_list:
                if y_seg is None or len(y_seg) < nperseg:
                    print(f"[WARN] Segment '{label}' trial is too short for nperseg={nperseg}. Skipping.")
                    continue
                # Removes the DC offset before welch 
                y_seg = y_seg - np.mean(y_seg)
                # Highpass filter to filter under 1hz artifacts
                y_seg = self.highpass_filter(y_seg, fs)
                
                f, p = welch(
                    y_seg,
                    fs=fs,
                    window="hamming",
                    nperseg=nperseg,
                    noverlap=int(nperseg * 0.75),
                    scaling="density"
                )

                p *= 1e12  # Convert V²/Hz → µV²/Hz

                if freqs is None:
                    freqs = f
                psds.append(p)

            if not psds:
                continue

            psds = np.array(psds)
            mean_psd = np.mean(psds, axis=0)
            sem_psd = np.std(psds, axis=0) / np.sqrt(psds.shape[0])  # SEM

            # Limit to max_freq
            valid = freqs <= max_freq
            freqs = freqs[valid]
            mean_psd = mean_psd[valid]
            sem_psd = sem_psd[valid]

            # Scale PSD values to match desired unit display (μV² × 10⁻²)
            mean_psd *= scale_factor
            sem_psd *= scale_factor

            results[label] = {
                "freqs": freqs,
                "mean": mean_psd,
                "sem": sem_psd
            }

        return results
    
    # Share Value to apply 50/60Hz filter in heatmap and power density
    @classmethod
    def clear(cls):
        cls.filtered_time = None
        cls.filtered_signal = None
        cls.filtered_fs = None
    
    @classmethod
    def store(cls, time, signal, fs):
        cls.filtered_time = time
        cls.filtered_signal = signal
        cls.filtered_fs = fs
        
    @classmethod
    def is_available(cls):
        return cls.filtered_signal is not None and cls.filtered_time is not None and cls.filtered_fs is not None
    
    @classmethod
    def get_stored_segments(cls):
        return cls.stored_segments