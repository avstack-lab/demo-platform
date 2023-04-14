


class _Controller():
    def __init__(self) -> None:
        self.processes = []

    def start(self):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError
    
    def end(self):
        for process in self.processes:
            process.join(timeout=0)
            process.kill()