from abc import ABC, abstractmethod

from src.ms_data import MSData, Resistance


class Model(ABC):
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def predict(self, data:MSData) -> Resistance:
        pass