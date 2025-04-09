import sys
import os
import time


# Record console output to a file
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def set_log(logname, log_path = './Logs/'):
    # Customize the directory to store log files
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Set the log file name according to the program runtime
    log_file_name = log_path + 'log-' + logname + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    checkPath_mkdirs(log_file_name)
    # Record normal print information
    sys.stdout = Logger(log_file_name)
    # Record traceback exception information
    sys.stderr = Logger(log_file_name)

if __name__ == '__main__':
    # Customize the directory to store log files
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Set the log file name according to the program runtime
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # Record normal print information
    sys.stdout = Logger(log_file_name)
    # Record traceback exception information
    sys.stderr = Logger(log_file_name)

def checkPath_mkdirs(path):
    p, n = os.path.split(path)
    if not os.path.exists(p):
        os.makedirs(p)
