import numpy as np

__all__ = ['Module', 'Dropout']

class Module:
    def __init__(self):
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def __setstate__(self, state):
        """Восстанавливает состояние объекта из словаря."""
        self.__dict__.update(state)

    def __getstate__(self):
        """Возвращает состояние объекта для сериализации."""
        return self.__dict__


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        mask = self.rng.random(x.shape) > self.p
        return np.where(mask, x / (1.0 - self.p), 0.0)