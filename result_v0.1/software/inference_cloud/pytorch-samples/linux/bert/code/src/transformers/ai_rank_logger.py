import os
import time
import logging

LOG_PATH = '../../log'
LOG_OFFLINE = 'offline_ips.log'
LOG_ONLINE = 'online_ips.log'
LOG_ACCU = 'accuracy_check.log'
logging.basicConfig(level = logging.INFO)

class ai_logger(object):
    def __init__(self):
        if os.path.exists(LOG_PATH):
            pass
        else:
            os.mkdir(LOG_PATH)
        self.offline_logger = logging.getLogger("offline_ips")
        self.offline_logger.propagate = False
        offline_hd = logging.FileHandler("%s/%s" % (LOG_PATH, LOG_OFFLINE))
        #rank_format = logging.Formatter()
        #offline_hd.setFormatter()
        self.offline_logger.addHandler(offline_hd)
        self.prefix = "AI-Rank-log "
        self.enable = True

    def disable(self):
        self.enable = False

    def info(self, msg):
        if self.enable:
            self.offline_logger.info(self.prefix + str(time.time()) +' '+ str(msg))