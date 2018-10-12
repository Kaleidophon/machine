import os

from seq2seq.util import LogCollection


if __name__ == "__main__":
    LOG_PATH = "../models/"
    lc = LogCollection()

    for subdir, dirs, _ in os.walk(LOG_PATH):
        for dir in dirs:
            model_path = os.path.join(LOG_PATH, dir)
            lc.add_log_from_folder(model_path)

