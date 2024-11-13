import threading
import time
from functions.graph_plot import get_plot_data

class DataFetcher:
    def __init__(self):
        self.thread = None
        self.last_fetch_time = 0
        self.fetch_interval = 3600  # 1 hour in seconds

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        while True:
            current_time = time.time()
            if current_time - self.last_fetch_time >= self.fetch_interval:
                print("Fetching data in background...")
                get_plot_data()
                self.last_fetch_time = current_time
            time.sleep(60)  # Check every minute

data_fetcher = DataFetcher()
