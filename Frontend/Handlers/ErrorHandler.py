class ErrorHandler:
    def __init__(self, logger_widget):
        self.logger = logger_widget
        
    def handle_error(self, error_msg):
        # Combine level + message into one string
        self.logger.AppendMessage(f"[Error] {error_msg}")

