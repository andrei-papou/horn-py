from typing import List, Dict, Callable

import numpy as np
from keras import Model, backend, activations
from keras.layers import Layer, Dense

from constants import LayerType
from data_encoder import DataEncoder


def get_layer_type(layer: Layer) -> str:
    return layer.__class__.__name__


class ModelSaver:

    def __init__(self):
        self._weight_id = 0
        self._data_encoder = DataEncoder()

    def _get_weight_id(self) -> int:
        self._weight_id += 1
        return self._weight_id

    def _save_weight(self, w: np.ndarray) -> int:
        wid = self._get_weight_id()
        self._data_encoder.add_tensor_entry((wid, w))
        return wid

    def _save_activation(self, name: str, func: Callable) -> Dict:
        return {
            'name': name,
            'type': LayerType.ACTIVATION.value,
            'func': func.__name__,
        }

    def _save_dense(self, layer: Dense) -> List[Dict]:
        weights: List[np.ndarray] = backend.batch_get_value(layer.weights)
        layer_dicts = [
            {
                'name': layer.name,
                'type': LayerType.DENSE.value,
                'weights_shape': weights[0].shape,
                'weights_id': self._save_weight(weights[0]),
                'bias_shape': weights[1].shape if layer.use_bias else None,
                'bias_id': self._save_weight(weights[1]) if layer.use_bias else None,
            }
        ]
        if layer.activation is not None and layer.activation is not activations.linear:
            layer_dicts.append(self._save_activation(name=f'{layer.name}__activation', func=layer.activation))
        return layer_dicts

    def save_model(self, model: Model, file_path: 'str'):
        layer_dicts = []
        layers: List[Layer] = model.layers
        for l in layers:
            lt = get_layer_type(l)
            if lt == LayerType.DENSE.value:
                layer_dicts.extend(self._save_dense(l))
        self._data_encoder.add_header_entry({
            'name': model.name,
            'layers': layer_dicts
        })

        with open(file_path, 'wb') as f:
            f.write(self._data_encoder.encode())
