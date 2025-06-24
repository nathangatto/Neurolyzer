from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFileSystemModel, QTreeView, QPushButton, QFileDialog, QStackedLayout
from PyQt5.QtCore import QSettings, QSortFilterProxyModel, QModelIndex, Qt
import os
import subprocess
import sys

class DirectoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 400, 100)
        self.settings = QSettings("WUSoftware", "Neurolyzer")

        self.proxymodel = QSortFilterProxyModel()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Select Directory Button
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.clicked.connect(self.selectDirectory)
        layout.addWidget(self.select_dir_button)

        # Dummy label for when no directory is selected
        self.dummyLabel = QLabel("No directory selected")
        self.dummyLabel.setStyleSheet("color: gray; font-style: italic; padding: 20px;")
        self.Open_Directory = QPushButton("Select Directory")

        # File system view
        self.DirectoryWindow = QFileSystemModel()
        self.DirectoryWindow.setRootPath('')
        self.DirectoryView = QTreeView()
        self.DirectoryView.setModel(self.DirectoryWindow)
        self.proxymodel.setSourceModel(self.DirectoryWindow)
        self.proxymodel.setSortRole(Qt.DisplayRole)
        self.DirectoryView.setModel(self.proxymodel)
        
        self.DirectoryView.doubleClicked.connect(self.onFileDoubleClicked)
        self.setMinimumSize(357, 100)
        
        def custom_less_than(self, left: QModelIndex, right: QModelIndex):
            source_model = self.sourceModel()
            left_type = source_model.data(source_model.index(left.row(), 2, left.parent()))
            right_type = source_model.data(source_model.index(right.row(), 2, right.parent()))
            
            preferredExtension = ['csv', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'svg', 'pkl']
            
            def priority(ext):
                ext = ext.lower()
                return 0 if any(p in ext for p in preferredExtension) else 1

            # Compare by priority first, then alphabetically
            prio_left = priority(left_type)
            prio_right = priority(right_type)

            if prio_left != prio_right:
                return prio_left < prio_right
            return left_type.lower() < right_type.lower()
        
        self.proxymodel.lessThan = custom_less_than.__get__(self.proxymodel)
        
        # Sets up the directory table
        self.DirectoryView.setColumnWidth(0, 230)
        self.DirectoryView.hideColumn(1)
        self.DirectoryView.setColumnWidth(2, 40)
        self.DirectoryView.sortByColumn(2, Qt.AscendingOrder)
        self.DirectoryView.hideColumn(3)

        # Stack layout for switching between dummy and directory views
        self.stack = QStackedLayout()
        self.stack.addWidget(self.dummyLabel)        # index 0
        # self.stack.addWidget(self.Open_Directory)
        self.stack.addWidget(self.DirectoryView)     # index 1

        self.go_back_button = QPushButton("Go Back")
        self.go_back_button.clicked.connect(self.goBackToDummy)

        # Add widgets to layout
        layout.addWidget(self.go_back_button)
        layout.addLayout(self.stack)

        self.setMinimumSize(357, 100)
        
        self.current_directory = None  # Add this

        # Start in dummy state
        self.showDummyPage()
        
        saved_path = self.settings.value('application/Directory_Path', type=str)
        if saved_path:
            self.setDirectoryPath(saved_path)
    
    def select_directory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if selected_dir:
            self.setDirectoryPath(selected_dir)
            self.settings.setValue('application/Directory_Path', selected_dir)

    def setDirectoryPath(self, directory_path):
        if directory_path and os.path.isdir(directory_path):
            self.current_directory = directory_path
            self.DirectoryWindow.setRootPath(directory_path)
            proxy_index = self.proxymodel.mapFromSource(self.DirectoryWindow.index(directory_path))
            self.DirectoryView.setRootIndex(proxy_index)
            self.stack.setCurrentIndex(1)
            self.go_back_button.show()
            self.select_dir_button.hide()
        else:
            self.showDummyPage()
    
    def selectDirectory(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if selected_dir:
            self.setDirectoryPath(selected_dir)
            self.settings.setValue('application/Directory_Path', selected_dir)

    def goBackToDummy(self):
        self.current_directory = None
        self.settings.remove('application/Directory_Path')
        self.showDummyPage()
        
    def showDummyPage(self):
        self.stack.setCurrentIndex(0)
        self.go_back_button.hide()
        self.select_dir_button.show()
        
    def onFileDoubleClicked(self, index: QModelIndex):
        source_index = self.proxymodel.mapToSource(index)
        file_path = self.DirectoryWindow.filePath(source_index)
        if os.path.isfile(file_path):
            print(f"Opening file: {file_path}")
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
    
    # def setDirectoryPath(self, directory_path):
    #     # Updates the old directory Path
    #     self.directory_path = directory_path
    #     self.DirectoryWindow.setRootPath(directory_path)
    #     self.DirectoryView.setRootIndex(self.DirectoryWindow.index(directory_path))

        
#         self.setMinimumSize(357, 200)
        
#         self.setLayout(layout)
    
#     def setDirectoryPath(self, directory_path):
#         # Updates the old directory Path
#         self.directory_path = directory_path
#         self.DirectoryWindow.setRootPath(directory_path)
#         self.DirectoryView.setRootIndex(self.DirectoryWindow.index(directory_path))