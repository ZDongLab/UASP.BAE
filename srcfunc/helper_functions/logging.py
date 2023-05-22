from datetime import datetime
from pathlib import Path
import logging
import os

def setup_logging(save_dir="./outputs/multi/log/", run_id='None', namemask="multilabel", default_level=logging.INFO):

    log_dir = Path(save_dir)
    if run_id == 'None':
        run_id = datetime.now().strftime(r'%y%m%d_%H%M%S')
    #_log_dir = log_dir / run_id 
    log_dir.mkdir(parents=True, exist_ok=True)
    log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }
            
    logger = logging.getLogger("testing")
    logger.setLevel(log_levels[1])
    logprint = logging.FileHandler(os.path.join(log_dir,run_id+"{namemk}.log".format(namemk=namemask)),"a",encoding="utf-8")
    logprint.setLevel(log_levels[1])
    formatter = logging.Formatter("%(asctime)s - %(filename)s-line:%(lineno)d - %(levelname)s - %(message)s")
    logprint.setFormatter(formatter)
    logger.addHandler(logprint)
    return logger