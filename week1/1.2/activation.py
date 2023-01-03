from abc import ABC, abstractmethod

class ActivationFunction(ABC):
   @abstractmethod
   def forward(self, x):
      pass
   @abstractmethod
   def gradient(self, x):
      pass
