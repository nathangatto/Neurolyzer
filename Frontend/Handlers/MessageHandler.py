class MessageHandler:
    def __init__(self, logger_widget, controller):
        self.logger = logger_widget
        self.controller = controller
        
    def cleanup_after_thread(self):
        # self.controller.clear_signal("result")
        # self.controller.clear_signal("error")
        # self.controller.clear_signal("finished")
        self.logger.AppendMessage("[MESSAGE] Thread finished and cleaned up.")