import os
import time
import logging
import hashlib

LOG_PATH = '../log'
LOG_OFFLINE = 'offline_ips.log'
LOG_ONLINE = 'online_ips.log'
logging.basicConfig(level = logging.INFO)

class logger(object):
    def __init__(self):
        if os.path.exists(LOG_PATH):
            pass
        else:
            os.mkdir(LOG_PATH)
        self.offline_logger = logging.getLogger("offline_ips")
        offline_hd = logging.FileHandler("%s/%s" % (LOG_PATH, LOG_OFFLINE))
        #rank_format = logging.Formatter()
        #offline_hd.setFormatter()
        self.offline_logger.addHandler(offline_hd)
        self.prefix = "AI-Rank-log "
    def info(self, msg):
        self.offline_logger.info(self.prefix + str(time.time()) +' '+ str(msg))

def log_md5(path, md5logger):
    # validation set
    for dirs in os.listdir(path):
        category = os.path.join(path, dirs)
        if os.path.isdir(category):
            for imgs in os.listdir(category):
                myhash = hashlib.md5()
                imgpath = os.path.join(category, imgs)
                f = open(imgpath,'rb')
                while True:
                    b = f.read(8096)
                    if not b:
                        f.close()
                        break
                    myhash.update(b)
                md5logger.info('{} md5: {}'.format(imgs, myhash.hexdigest()))