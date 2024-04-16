import os


def paths():

    if os.name == 'nt':
        return [
                os.path.join('\\users', 'Bill', 'Documents', 'BEAR_DOCUMENTS', 'Work Stuff', 'Resume'),   # WINDOWS
                os.path.join('\\users', 'Bill', 'Desktop'),                             # WINDOWS
                f"MANUAL ENTRY"                                                         # WINDOWS / LINUX
        ]
    elif os.name == 'posix':
        return [
                os.path.join('/media', 'bill', 'SANDISK 64GB'),  # LINUX
                os.path.join('/home', 'bill', 'Desktop'),        # LINUX
                os.path.join('/media', 'bear', 'SANDISK 64GB'),  # LINUX
                os.path.join('/home', 'bear', 'Desktop'),        # LINUX
                f"MANUAL ENTRY"                                  # WINDOWS / LINUX
        ]






