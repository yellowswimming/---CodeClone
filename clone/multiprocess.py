from multiprocessing import Process, Queue, freeze_support
import search

def mul_search(s):
    result = {}
    queue = Queue()

    PROCESS_NUM = 4

    processes = []
    freeze_support()
    
    # create PROCESS_NUM processes
    for i in range(PROCESS_NUM):
        p = Process(target=search.search, args=(s,1/PROCESS_NUM,i,queue),daemon=True)
        processes.append(p)

    # start all processes
    for p in processes:
        p.start()

    for i in range(PROCESS_NUM):
        result.update(queue.get())

    return result
    
def result_mulsearch(s):
    r = mul_search(s)
    if r != None:
        return r