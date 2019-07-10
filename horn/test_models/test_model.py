from typing import Optional

import numpy as np
from keras.models import load_model, Model


class TestModel:

    def __init__(self, load_from: str):
        self._model: Optional[Model] = load_model(load_from) if load_from is not None else None

    def train_model(self, save_to: str):
        raise NotImplementedError()

    def generate_test_samples(self, n: int) -> (np.ndarray, np.ndarray):
        raise NotImplementedError()
