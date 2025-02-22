from abc import ABC, abstractmethod

class ProcessingMethod(ABC):
    def __init__(self):
        self.name = None

    @abstractmethod
    def process(self, T, num_samples):
        pass
