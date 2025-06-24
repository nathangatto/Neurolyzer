import pyabf
import hashlib
import time
import datetime
import logging
    
class FileSession:
    def __init__(self, abf):
        self.abf = abf
        # File Info determining 
        self.DataRate = None
        self.SweepLengthSec = None
        self.SweepCount = None
        self.ChannelCount = None
        self.Protocol = None
    
    def PlotRawABF(self):
        LoadedFile = self.abf

        x = y = current = None

        if LoadedFile.channelCount >= 1:
            LoadedFile.setSweep(sweepNumber=0, channel=0)
            x = LoadedFile.sweepX
            y = LoadedFile.sweepY.copy()

        if LoadedFile.channelCount >= 2:
            LoadedFile.setSweep(sweepNumber=0, channel=1)
            current = LoadedFile.sweepY.copy()

        return x, y, current
    
    def GetFileInfo(self):
        self.DataRate = self.abf.dataRate
        print(f"Sampling rate: {self.abf.dataRate} Hz")
        self.SweepLengthSec = self.abf.sweepLengthSec
        print(f"Duration: {self.abf.sweepLengthSec:.2f} seconds")
        self.SweepCount = self.abf.sweepCount
        print(f"Number of sweeps: {self.abf.sweepCount}")
        self.ChannelCount = self.abf.channelCount
        print(f"Channels: {self.abf.channelCount}")
        self.Protocol = self.abf.protocol 
        print(f"Protocol: {self.abf.protocol}")
        
    def DetemineAnalysisType(self):
        # Map for all possibilities
        SelectedAnalysis = ""
        keyword_map = {
            "K+": "K+ Clearance",
            "K puff": "K+ Clearance",
            "Chirp": "Resonance Frequency",
            "Alba oscillations": "Extracellular",
            "oscillation": "Extracellular",
        }
        
        for keyword, analysis_type in keyword_map.items():
            if keyword in self.Protocol:
                SelectedAnalysis = analysis_type 
        
        if self.ChannelCount > 1 and (SelectedAnalysis == "Extracellular" or "Resonance Frequency"):
            return SelectedAnalysis
        elif self.ChannelCount == 1 and SelectedAnalysis == "K+ Clearance":
            return SelectedAnalysis
        else:
            pass

    def calculate_sha256(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
        

def FileLoader(file_path): 
    abf = pyabf.ABF(file_path)
    pre_hash = FileSession.calculate_sha256(file_path)
    post_hash = FileSession.calculate_sha256(file_path)
    match = pre_hash == post_hash
    print("Channel Count:", abf.channelCount)
    print("Channel Names:", abf.channelList)
    print("Protocol:", abf.protocol)
    print("Sampling Rate:", abf.dataRate)
    if match:
        print("MATCHED")
    return FileSession(abf)


def log_blocked_write(file_path, reason="Write attempt blocked"):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_msg = f"{timestamp} BLOCKED: {reason} on {file_path}"
    print(log_msg)
    logging.warning(log_msg)

def ExportToPath(self, path):
    if path.endswith(".abf"):
        log_blocked_write(path, "Refused to overwrite ABF file")
        return