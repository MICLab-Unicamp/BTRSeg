import subprocess
import os
import logging
from datetime import datetime


def get_git_hash(path="/home/diedre/git/diedre_phd"):
    if os.path.isdir(path):
        wd = os.getcwd()
        try:
            os.chdir(path)
            output = subprocess.run(["git", "rev-parse", "--short=4", "..."], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            output = output.stdout.decode('UTF-8')[:4]
        except Exception as e:
            logging.warning(f"Failed to get git commit version: {e}")
        finally:
            os.chdir(wd)

        return output
    else:
        logging.warning("Directory for git hash versioning doesnt exist. Returning date version.")
        return str(datetime.now())
