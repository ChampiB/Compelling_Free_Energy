from datetime import datetime


#
# This class can be used to estimate simulation runtime.
#
class TimeTracker:

    def __init__(self):
        self.start = datetime.now()
        self.stop = datetime.now()

    def tic(self):
        self.start = datetime.now()

    def toc(self):
        self.stop = datetime.now()

    def print(self, file):
        file.write("========== TIME TRACKER ==========\n")
        file.write("Running Time: " + str(self.stop - self.start))
