from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QFormLayout, QCheckBox, QPushButton, QFileDialog, QLineEdit, QComboBox, QApplication, QDockWidget
from PyQt5.QtCore import QSettings

class Settings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(100, 100, 600, 600)
        self.setModal(False)
        
        # Initialize QSettings
        self.settings = QSettings("WUSoftware", "Neurolyzer")
        print(f"Settings file: {self.settings.fileName()}")
        self.selected_folder = None

        # Layouts
        main_layout = QVBoxLayout()
        settings_tabs = QTabWidget()

        # Appearance Tab Creation 
        appearance_tab = QWidget()
        appearance_layout = QFormLayout()
        
        # Full Screen at start
        self.full_screen_checkbox = QCheckBox()
        appearance_layout.addRow("Full screen at start", self.full_screen_checkbox)
        
        # Select Wanted Themes
        self.theme_selection = QComboBox()  # Dropdown to select theme
        self.theme_selection.addItems(["Native", "Light Theme", "Dark Theme"])
        appearance_layout.addRow("Select Theme", self.theme_selection)
        appearance_tab.setLayout(appearance_layout)
        
        # Application Tab Creation
        application_tab = QWidget()
        application_layout = QFormLayout()
        
        # Directory Selection 
        directory_layout = QHBoxLayout()
        self.DirectoryPath = QLineEdit()
        self.DirectoryPath.setReadOnly(True)
        self.DirectoryPath.setText(self.selected_folder)
        self.DirectoryFolder = QPushButton("Select Folder")
        self.DirectoryFolder.clicked.connect(self.browse_folders)
        directory_layout.addWidget(self.DirectoryPath)
        directory_layout.addWidget(self.DirectoryFolder)
        application_layout.addRow("Directory Folder", directory_layout)
        
        application_tab.setLayout(application_layout)
        
        settings_tabs.addTab(appearance_tab, "Appearance")
        settings_tabs.addTab(application_tab, "Application")
        main_layout.addWidget(settings_tabs)

        # Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.applySettings)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # Load saved theme from settings file
        saved_theme = self.settings.value("appearance/theme", "Light Theme", type=str)
        self.theme_selection.setCurrentText(saved_theme)
        self.applyTheme(saved_theme)
        
        Saved_directory = self.settings.value("application/Directory_Path", self.selected_folder)
        self.DirectoryPath.setText(Saved_directory)
        self.parent().Directory.setDirectoryPath(Saved_directory)
        
    # Settings applied via Apply Button 
    def applySettings(self):
        # Save the state of the checkbox to settings
        selected_theme = self.theme_selection.currentText()
        self.settings.setValue("appearance/theme", selected_theme)
        self.applyTheme(selected_theme)
        self.settings.setValue('appearance/full_screen_start', self.full_screen_checkbox.isChecked())
        self.settings.setValue('application/Directory_Path', self.selected_folder)
        self.parent().Directory.setDirectoryPath(self.selected_folder)
        self.parent().restore_user_settings()

        # Applies Settings to the parent window immediately
        if self.parent():
            if self.full_screen_checkbox.isChecked():
                self.parent().showMaximized()
            else:
                self.parent().showNormal()

        self.accept()  # Close the dialog

    def applyTheme(self, theme):
        # Apply theme globally (to the entire application)
        app = QApplication.instance()  # Get the QApplication instance

        if theme == "Light Theme":
            app.setStyleSheet("""
                QWidget {
                    background-color: white;
                    color: black;
                }
                QPushButton {
                    background-color: #f0f0f0;
                    color: black;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QTabWidget {
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                }
                QTabBar {
                    qproperty-expanding: true;
                    background-color: #e0e0e0;
                    border: 1px solid #ccc;
                }
                QTabBar::tab {
                    padding: 5px;
                    margin: 0px;
                    background-color: #ffffff;
                    border: 1px solid #ccc;
                }
                QTabBar::tab:selected {
                    background-color: #d0d0d0;
                }
                QLineEdit {
                    background-color: #ffffff;
                    border: 1px solid #ccc;
                    color: black;
                }
                QComboBox {
                    background-color: white;
                    border: 1px solid #ccc;
                    color: black;
                }
                QComboBox::drop-down {
                    border: 1px solid #ccc;
                }
                QDockWidget::title {
                    background-color: #DADADA;
                    padding: 1px;
                    font-weight: bold;
                }
                QMenu::item{
                    background-color: transparent;
                    color: black
                }
                QMenu::item:selected {
                    background-color: #DADADA;
                    color: black;
                }
            """)
        elif theme == "Dark Theme":
            app.setStyleSheet("""
                QWidget {
                    background-color: #2E2E2E;
                    color: white;
                }
                QPushButton {
                    background-color: #333;
                    color: white;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #444;
                }
                QTabWidget {
                    background-color: #444;
                    border: 1px solid #666;
                }
                QTabBar {
                    background-color: #555;
                    border: 1px solid #666;
                    qproperty-expanding: true;
                }
                QTabBar::tab {
                    padding: 5px;
                    margin: 0px;
                    background-color: #333;
                }
                QTabBar::tab:selected {
                    background-color: #666;
                }
                QLineEdit {
                    background-color: #333;
                    border: 1px solid #555;
                    color: white;
                }
                QTextEdit {
                    background-color: #333;
                    border: 1px solid #555;
                    color: white;
                }
                QComboBox {
                    background-color: #333;
                    border: 1px solid #555;
                    color: white;
                }
                QComboBox::drop-down {
                    border: 1px solid #555;
                }
                QDockWidget::title {
                    background-color: #4A4A4A;
                    padding: 1px;
                    font-weight: bold;
                }
                QMenu::item{
                    background-color: transparent;
                    color: white;
                }
                QMenu::item:selected {
                    background-color: #4A4A4A;
                    color: white;
                }
            """)
        elif theme == "Native":
        # ❗️Clear all styles to return to Qt's native default
            app.setStyleSheet("""
                QDockWidget::title {
                    background-color: #DADADA;
                    padding: 1px;
                    font-weight: bold;
                }
            """)
            self.theme_selection.setCurrentText("Native")
            
        for dock in self.findChildren(QDockWidget):
            dock.style().unpolish(dock)
            dock.style().polish(dock)
            dock.update()
            
    def showEvent(self, event):
        super().showEvent(event)
        self.centerWindow()

        # Load saved setting
        full_screen = self.settings.value('appearance/full_screen_start', False, type=bool)
        self.full_screen_checkbox.setChecked(full_screen)

    def centerWindow(self):
        if self.parent():
            parent_center = self.parent().frameGeometry().center()
            qr = self.frameGeometry()
            qr.moveCenter(parent_center)
            self.move(qr.topLeft())

    # For Importing files
    def browse_folders(self):
        # Opens file directory to select ABF file
        folder_path = QFileDialog.getExistingDirectory(self, caption= "Select Folder", directory= "", options=QFileDialog.ShowDirsOnly)
        # # Adds file if selected
        if folder_path:
            # saved variable in Self instance
            self.selected_folder = folder_path  
            self.DirectoryPath.setText(self.selected_folder)    
        else:
            print("No file selected")