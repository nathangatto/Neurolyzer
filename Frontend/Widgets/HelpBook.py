from PyQt5.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QFormLayout, QLineEdit, QPushButton, QListWidget, QTextEdit, QLabel
)
from PyQt5.QtCore import QSettings, Qt

class HelpBook(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(100, 100, 800, 500)
        self.setWindowTitle("Neurolyzer Help Book")

        # ─── Main layout ───
        Main_layout = QVBoxLayout()

        # ─── Help tabs ───
        self.Help_tabs = QTabWidget()

        # ─── General Help Tab ───
        self.GenHelp_tab = QWidget()
        GenHelpTab_Layout = QVBoxLayout()

        header = QLabel("📖 Select a Help Topic")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        GenHelpTab_Layout.addWidget(header)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search topics...")
        self.search_input.textChanged.connect(self.filter_topics)
        GenHelpTab_Layout.addWidget(self.search_input)

        content_layout = QHBoxLayout()
        self.topic_list = QListWidget()
        self.topic_list.addItems([
            "▶ General Overview",
            "   Getting Started",
            "   Toolbar",
            "▶ Tools & Controls",
            "   Extracellular Analysis",
            "   Resonance Frequency",
            "   K+ Clearance",
            "▶ Plot Area",
            "   Plot Area",
            "▶ Console & Results",
            "   Results and Console",
            "▶ Directory",
            "   Directory",
            "▶ Shortcuts",
            "   Save Session",
            "   Load Session",
            "   Exporting and Saving"
            "   Settings",
            "▶ Extra",
            "   UI and Docking",
            "   Status Bar",
            "   Troubleshooting",
            "   About",
        ])

        self.topic_list.currentItemChanged.connect(self.display_help_text)
        content_layout.addWidget(self.topic_list, 1)

        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        content_layout.addWidget(self.help_text, 3)

        GenHelpTab_Layout.addLayout(content_layout)
        self.GenHelp_tab.setLayout(GenHelpTab_Layout)

        self.Help_tabs.addTab(self.GenHelp_tab, "General")

        SetLayout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        SetLayout.addStretch()
        SetLayout.addWidget(close_button)
        Main_layout.addWidget(self.Help_tabs)
        Main_layout.addLayout(SetLayout)
        self.setLayout(Main_layout)
        
        self.help_content = {
            "Getting Started": (
                "Welcome to Neurolyzer! This tool helps you load, analyze, and visualise ABF files with ease.\n"
                "Supported on Windows and macOS. No internet connection required for basic use."
            ),
            
            "Toolbar": (
                "The top menu bar contains:\n"
                "• File: Open ABF File, Export as CSV, Save Session, Export Results, Quit.\n"
                "• Tools: Access Settings, Clear Console, Restart the App.\n"
                "• Help: User Guide (F1) and About.\n\n"
                "Each menu option has an assigned shortcut for ease of access."
            ),

            #"Loading ABF Files": "Use 'File > Open ABF File' to load your neurophysiology data. Only `.abf` format is supported. Once loaded, the experiment type can be selected for analysis.",

            "Extracellular Analysis": (
                "This mode allows power spectrum, and power density visualisation. "
                "You can segment specific response phases and apply a noise filter to clean up signal noise. "
                "Graphs generated include channel0, channel1, heatmaps, and power spectrum."
            ),

            "Resonance Frequency": (
                "This module provides tools to generate ZAP profiles, compute resonance frequencies, and calculate Half-Width at Attenuation (HWA). "
                "After selecting the Resonance tab, run ZAP profiling to view impedance across frequencies, identify peak resonance, and visualize HWA with overlaid markers."
            ),
                
            "K+ Clearance": (
                "This tool segments episodes from the ABF file and calculates Tau and Rate Constants to measure potassium clearance dynamics.\n\n"
                "Calibration Parameters:\n"
                "To convert from measured voltage (mV) to actual concentration (mM), calibrate your ion-selective microelectrode. Two key parameters control this conversion:\n\n"
                "• Electrode slope (DEFAULT_SLOPE):\n"
                "    - Definition: Sensitivity of your K+-sensitive electrode, in millivolts per decade change in concentration.\n"
                "    - Typical range: 55–60 mV/decade at room temperature.\n"
                "    - Role: [K+] = 10^((V - E0) / slope), where V is measured voltage, E0 is the zero-point potential.\n"
                "    - Why change it: Each electrode is different. After recording voltages in known K+ concentrations, fit a line of voltage vs log[K+] and enter its slope here for accurate conversion.\n\n"
                "• Resting K+ concentration (RESTING_K_MM):\n"
                "    - Definition: The known extracellular potassium concentration (in mM) in your perfusion solution before stimulation.\n"
                "    - Role: Used to find the calibration intercept: Vbaseline = (slope × log10[K+]rest)\n"
                "    - Why change it: Different labs/protocols may use 2.5, 3, 5 mM, etc. Matching this ensures correct conversion to true extracellular K+ concentration."
            ),
            
            "Plot Area": (
                "The Plot Area is the central space of the Neurolyzer interface where all visual data is displayed."
                "Depending on the selected analysis type, the following plots can appear:\n"
                
                "• Raw ABF – Displays channel 0 and channel 1 voltage traces\n"
                "• FFT Plot – Shows frequency distribution (Fourier transform)\n"
                "• Spectrum Plot – Plots Power Spectral Density (PSD) by segment\n"
                "• Heatmaps – Visualise power density over time-frequency space\n"
                "• Zap Profile – Impedance curve from resonance frequency data\n"
                "• Segments (Seg 1–4) – Individual fits for K⁺ clearance analysis\n"
                
                "Each graph appears as a tab in the plot area. You can switch between them by clicking the tab headers. "
                "Graphs can be exported individually or all at once using the Export Results button in the sidebar.\n"
                
                "Note: Some graphs only appear after running their associated analysis (e.g., FFT, segmentation, etc.)."
            ),

            "Results and Console": (
                "The bottom panel has two tabs:\n"
                "• Results: Displays key findings like resonance frequency, Tau, PSDs, and more.\n"
                "• Console: Shows system messages, errors, and manual logs.\n"
                "Use the console to monitor analysis progress and debug output."
            ),
            
            "Directory": (
                "The Directory section is used to navigate and select a local folder."
                "This folder serves as a default path for opening and saving ABF files or exported images."
                "Use the 'Select Directory' button to browse folders."
                "This path may also be stored in the application settings for future use."
            ),
            
            "Load Session": (
                "Use 'Load Session' to reload a previously saved analysis session.\n"
                "This restores:"
                "• Plotted graphs (Raw ABF, Segments)"
                "• Results and messages"
                "• Analysis type (Extracellular / Resonance / K⁺)"
                
                "Note: Only sessions saved using Neurolyzer's own 'Save Session' feature are compatible."
            ),
            
            "Save Session": (
                "Use 'Save Session' to store your current analysis:"
                "• Raw ABF plots"
                "• Segment results"
                "• Heatmaps, impedance, spectrum plots"
                "• Text results and log"
                
                "Sessions are saved locally. These can later be reloaded via 'Load Session' to continue your work."
            ),    
            
            "Exporting and Saving": (
                "Use the Export Results button to save outputs:\n"
                "• Graphs: Save as PNG images.\n"
                "• Results: Choose TXT or CSV formats for compatibility with Excel.\n"
                "Choose 'Export All' to save everything at once, including all graphs and results."
            ),
            
            "Settings": (
                "The Settings dialog includes:\n\n"
                "Appearance tab:\n"
                "• Toggle full-screen on startup\n"
                "• Select your preferred UI theme (Native, Fusion, etc)\n\n"
                "Application tab:\n"
                "• Set a default directory for file browsing.\n\n"
                "Press Apply to save changes or Close to discard them."
            ),

            "UI and Docking": (
                "Widgets are fully dockable. Rearrange by dragging panel headers. "
                "Use the toolbar or View menu to reset layout. You can resize areas with draggable dividers and float or minimise sections for custom workspace preferences."
            ),
            
            "Status Bar": (
                "Located at the bottom edge of the application.\n\n"
                "• Left Side: Displays the most recent log message (e.g., 'File Loaded').\n"
                "• Right Side: Shows whether an ABF file is currently loaded.\n\n"
                "It updates automatically based on analysis and system activity."
            ),

            "Troubleshooting": (
                "Common issues:\n"
                "• File not loading: Ensure it’s a valid `.abf` file.\n"
                "• Graphs not visible: Try resizing the window or checking the Plot Area.\n"
                "• Export errors: Make sure an output folder is selected.\n"
                "For persistent issues, check console logs or contact the development team."
            ),

            "About": (
                "Neurolyzer v1.0\n"
                "Developed by the PX Group from Western Sydney University for the Cellular Neurophysiology Lab.\n"
                "Modular, offline-capable ABF analysis and visualisation platform.\n"
                "For research and educational use."
            )
        }

    def showEvent(self, event):
        super().showEvent(event)
        self.centerWindow()

    def centerWindow(self):
        if self.parent():
            center = self.parent().frameGeometry().center()
            geo = self.frameGeometry()
            geo.moveCenter(center)
            self.move(geo.topLeft())

    def filter_topics(self):
        text = self.search_input.text().lower()
        for i in range(self.topic_list.count()):
            item = self.topic_list.item(i)
            item.setHidden(text not in item.text().lower())
        self.search_help_text(text)
    
    def search_help_text(self, keyword):
        # Optional: Search within content as well
        keyword = keyword.strip().lower()
        for key, content in self.help_content.items():
            if keyword in key.lower() or keyword in content.lower():
                self.help_text.setText(content)
                return

    def display_help_text(self):
        current_item = self.topic_list.currentItem()
        if not current_item:
            return
    
        topic = current_item.text().strip()
        content = self.help_content.get(topic, "No help available for this topic.")
        self.help_text.setText(content)
    
        # Ignore section headers
        if topic.startswith("▶") or topic.startswith("–"):
            self.help_text.setText("This is a section header. Please select a topic below.")
            return
