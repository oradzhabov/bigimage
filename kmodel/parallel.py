# from multiprocessing import Pool
import threading
import queue


class SimpleProcessor(object):
    def __init__(self, func, args, pool):
        self.func = func
        self.args = args
        self.workers = 0
        self.pool = pool
        self.queue = None
        self.stop_signal = None
        self.running_signal = None
        self.run_thread = None

    def start(self, max_queue_size=10):
        if not self.is_running():
            # self.pool = Pool(workers)
            self.queue = queue.Queue(max_queue_size)
            self.stop_signal = threading.Event()
            self.running_signal = threading.Event()
            self.run_thread = threading.Thread(target=self._run)
            self.run_thread.daemon = True
            self.run_thread.start()

    def _run(self,):
        it = iter(self.args)
        for arg in it:
            if self.stop_signal.is_set():
                break
            if not isinstance(arg, tuple):
                arg = (arg, )
            future = self.pool.apply_async(self.func, arg)
            self.queue.put(future, block=True)
        self.running_signal.set()
        # self.pool.close()
        # self.pool.join()
    
    def stop(self, timeout=None):
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
    
    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()
    
    def get(self, timeout=30):
        try:
            while not self.running_signal.is_set() or self.queue.unfinished_tasks != 0:
                future = self.queue.get(block=True)
                inputs = future.get(timeout=timeout)
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
            self.stop()
        except Exception as _:
            self.stop()
            raise
            
    def __del__(self):
        if self.is_running():
            self.stop()
            # self.pool.close()
            # self.pool.join()
