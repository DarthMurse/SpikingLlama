from slim_download import *
from star_download import *
from book_download import *
from time import sleep
from multiprocessing import Process
import sys

if __name__ == '__main__':
    if sys.argv[1] == "slim":
        fn = slim_download
    elif sys.argv[1] == "star":
        fn = star_download
    elif sys.argv[1] == "book":
        fn = book_download
    else:
        print("dataset name not valid!")

    procs = []
    procs.append(Process(target=fn, args=()))
    procs[-1].start()
    procs[-1].join()
    while True:
        if not procs[-1].is_alive():
            procs.append(Process(target=fn, args=()))
            procs[-1].start()
            print("process{len(procs)} starts!")
            procs[-1].join()
            print("the process ends!")
            sleep(120)
