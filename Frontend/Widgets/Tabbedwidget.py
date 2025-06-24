from PyQt5.QtWidgets import QWidget, QTabWidget , QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QPushButton
from datetime import datetime
from PyQt5.QtGui import QFont

# Tabbed Results and console widget
class TabbedWindow(QWidget):
    def __init__(self, update_status_bar):
        super().__init__()
        #Sets dimensions for the window
        self.setGeometry(100,100,800,300)
        self.setMinimumSize(300,160)
        
        self.update_status_bar = update_status_bar
        
        # Vertical layout definition
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        #Creates Tabs
        self.ResultTab = QWidget()
        self.ConsoleTab = QWidget()
        #Adds it to the Qtabwidget
        self.tabs.addTab(self.ResultTab, "Results")
        self.tabs.addTab(self.ConsoleTab, "Console")
        
        # Overall Results Tab layout 
        self.ResultsTab_layout = QHBoxLayout()
        
        self.RButtons_layout = QVBoxLayout()
        # Buttons and their Functionality
        self.Results1 = QPushButton("Results 1")
        self.Results1.setStyleSheet("""
                                    QPushButton{
                                        border: None;
                                    }
                                    QPushButton::hover{
                                        background-color: #D3D3D3;
                                    }
                                    QPushButton::pressed{
                                        background-color: #D3D3D3;
                                    }
                                    """)
        
        self.ClearResults = QPushButton("Clear")
        self.ClearResults.clicked.connect(self.clear_results)
        
        self.RButtons_layout.addWidget(self.Results1)
        self.RButtons_layout.addStretch()
        self.RButtons_layout.addWidget(self.ClearResults)
        
        #Results tab Layout
        self.Results_layout = QVBoxLayout()
        
        #Creating layout elements and their rules
        self.ResultsBox = QTextEdit()
        self.ResultsBox.setReadOnly(True)
        # Use monospaced font for proper alignment
        self.ResultsBox.setFont(QFont("Fira Math", 10))
        
        self.Results_layout.addWidget(self.ResultsBox)
        
        #Adds elements to the widget in a Horisontal manner
        self.ResultsTab_layout.addLayout(self.Results_layout)
        self.ResultsTab_layout.addLayout(self.RButtons_layout)
        self.ResultTab.setLayout(self.ResultsTab_layout)
        
        #Console tab Layout
        self.ConsoleTab_layout = QHBoxLayout()
        
        #Creates Right side of the console, Includes buttons for functionality
        self.CButtons_layout = QVBoxLayout()
        # Buttons and their Functionality
        self.NewConsole = QPushButton("Reset")
        self.NewConsole.setShortcut("Ctrl+Shift+N")  
        self.NewConsole.clicked.connect(self.ClearMessage)
        
        self.EnterCommand = QPushButton("Enter")
        self.EnterCommand.clicked.connect(self.PrintLogMessage)
        
        self.CButtons_layout.addStretch()
        self.CButtons_layout.addWidget(self.NewConsole)
        self.CButtons_layout.addWidget(self.EnterCommand)
        
        #Creates Left side of the console, Includes command line and log box
        self.Console_layout = QVBoxLayout()
        self.ConsoleBox = QTextEdit()
        self.ConsoleBox.setReadOnly(True)
        self.InputBox = QLineEdit()
        self.InputBox.setPlaceholderText("Command Line, Type to use")
        self.InputBox.returnPressed.connect(self.EnterCommand.click)
        self.Console_layout.addWidget(self.ConsoleBox)
        self.Console_layout.addWidget(self.InputBox)
        
        #Merges all layouts into one
        self.ConsoleTab_layout.addLayout(self.Console_layout)
        self.ConsoleTab_layout.addLayout(self.CButtons_layout)
        self.ConsoleTab.setLayout(self.ConsoleTab_layout)
       
        #Puts everything together
        layout.addWidget(self.tabs)
        self.setLayout(layout)
   
    def AppendMessage(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {message}"
        self.ConsoleBox.append(full_msg)
    
        # Call the main window's update_status_bar method
        if callable(self.update_status_bar):
            self.update_status_bar(console_message=message)
        
    def AppendResults(self, results: str):
        self.ResultsBox.append(results)
    
    def ClearMessage(self):
        self.ConsoleBox.clear()

    def clear_all(self):
        """Erase both the Results and Console panes."""
        self.ResultsBox.clear()
        self.ConsoleBox.clear()

    def clear_results(self):
        """Erase only the Results pane."""
        self.ResultsBox.clear()

    def clear_console(self):
        """Erase only the Console pane."""
        self.ConsoleBox.clear()
    
    def PrintLogMessage(self):
        message = self.InputBox.text()
        self.ConsoleBox.append(message)  
        

