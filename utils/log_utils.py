import os
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
