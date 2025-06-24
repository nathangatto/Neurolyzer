import pyabf
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, windows

class ResonanceFreq:
    def __init__(self, filepath, FreqRange):
        #Import File path
        self.abf = pyabf.ABF(filepath)
        
        #Set ABF variables for plotting and analysis
        self.Time = None
        self.Voltage = None
        self.Current = None
        self.C_FFT = None
        self.V_FFT = None
        self.freqs = None
        self.pos_mask = None
        self.impedance = None
        self.ResFreq = None
        self.LenTime = None
        self.HWA = None
        
        self.FreqRange  = float(FreqRange)
        
        # Other Variables needed
        self.DataRate = self.abf.dataRate
        
    def highpass_filter(self, data, fs, cutoff=0.5, order=3):
        b, a = butter(order, cutoff / (0.5 * fs), btype='high')
        return filtfilt(b, a, data)
        
    def ExtractData(self):
        # Fetch time and voltage from channel 0
        self.abf.setSweep(sweepNumber= 0, channel = 0)
        self.Time = self.abf.sweepX
        self.Voltage = self.abf.sweepY.copy()
        # Fetch Current from channel 1 
        self.abf.setSweep(sweepNumber= 0, channel = 1)
        self.Current = self.abf.sweepY.copy()
        
        self.Voltage = self.highpass_filter(self.Voltage, self.DataRate)
        self.Current = self.highpass_filter(self.Current, self.DataRate)
        
        # DEBUG
        # print(f"Voltage Mean: {np.mean(self.Voltage):.3f}, Std: {np.std(self.Voltage):.3f}")
        # print(f"Current Mean: {np.mean(self.Current):.3f}, Std: {np.std(self.Current):.3f}")
    
    def CalcFFT(self):
        self.LenTime = len(self.Time)
        self.DT = self.Time[1] - self.Time[0]
        self.freqs = fftfreq(self.LenTime, self.DT)
        self.pos_mask = self.freqs > 0

        window = windows.hann(self.LenTime)
        voltage_win = self.Voltage * window
        current_win = self.Current * window

        self.V_FFT = fft(voltage_win)
        self.C_FFT = fft(current_win)


    def CalcImpedance(self):
        # Compute impedance magnitude |Z(f)| = |V(f)| / |I(f)|
        impedance_raw = np.abs(self.V_FFT[self.pos_mask]) / np.abs(self.C_FFT[self.pos_mask])
        self.finalFreq = self.freqs[self.pos_mask]

        # filter frequency range (e.g., 0–100 Hz)
        freq_filter = self.finalFreq < 100
        self.finalFreq = self.finalFreq[freq_filter]
        self.impedance = (impedance_raw[freq_filter]) * 1e6  # Convert to MΩ
        
        print(self.FreqRange)
        targetFreqs = np.arange(0, self.FreqRange + 0.5, 0.5)
        ImpedanceAtHalf = []
        
        for f in targetFreqs:
            index = (np.abs(self.finalFreq - f)).argmin()
            ImpedanceAtHalf.append(self.impedance[index])
            
        self.PlottingFreq = targetFreqs
        self.PlottingImp = np.array(ImpedanceAtHalf)
    
        return self.PlottingFreq, self.PlottingImp
    
    def CalcResonanceFreq(self):
        idx = np.argmax(self.PlottingImp)
        self.ResFreq = self.PlottingFreq[idx]
        return self.ResFreq
    
    def find_crossing(self, freqs, imp, target):
        freqs = np.asarray(freqs)
        imp = np.asarray(imp)

        # Find where the signal crosses the target value
        crossing_indices = []
        for i in range(len(imp) - 1):
            if (imp[i] - target) * (imp[i + 1] - target) < 0:
                crossing_indices.append(i)

        if not crossing_indices:
            # No crossing found, fallback to closest point
            idx_closest = int(np.argmin(np.abs(imp - target)))
            return float(freqs[idx_closest]), float(imp[idx_closest])

        # Use the first crossing and do linear interpolation
        i = crossing_indices[0]
        x0, x1 = freqs[i], freqs[i + 1]
        y0, y1 = imp[i], imp[i + 1]

        # Linear interpolation formula
        f_cross = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        return f_cross, target
    
    def CalcHWA(self):
        z_peak = np.max(self.PlottingImp)
        Target = z_peak / 2
        
        idx_peak = np.argmax(self.PlottingImp)
        
        left_freq = np.array(self.PlottingFreq[:idx_peak+1])
        left_imp = np.array(self.PlottingImp[:idx_peak+1])
        self.f_left, self.z_left = self.find_crossing(left_freq, left_imp, Target)
        
        right_freq = self.PlottingFreq[idx_peak:]
        right_imp = self.PlottingImp[idx_peak:]
        self.f_right, self.z_right = self.find_crossing(right_freq, right_imp, Target)
        
        self.HWA = self.f_right - self.f_left
        
        return self.f_left, self.f_right, self.z_left, self.z_right, self.HWA