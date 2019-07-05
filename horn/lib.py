from io import BytesIO
from typing import Dict, List, Callable

import numpy as np

from keras import backend as K, Model, activations
from keras.models import load_model
from keras.layers import Layer, Dense

from constants import LayerTypes


def _save_weight(w: np.ndarray) -> str:
    buf = BytesIO()
    np.save(buf, w)
    return buf.getvalue()


def _save_activation(name: str, func: Callable) -> Dict:
    return {
        'name': name,
        'type': LayerTypes.ACTIVATION.value,
        'func': func.__name__,
    }


def _save_dense(layer: Dense) -> List[Dict]:
    weights = K.batch_get_value(layer.weights)
    layer_dicts = [
        {
            'name': layer.name,
            'type': LayerTypes.DENSE.value,
            'weights_shape': weights[0],
            'bias_shape': weights[1] if layer.use_bias else None,
        }
    ]
    if layer.activation is not None and layer.activation is not activations.linear:
        layer_dicts.append(_save_activation(name=f'{layer.name}__activation', func=layer.activation))
    return layer_dicts


def save_model(model: Model):
    layer_dicts = []
    layers: List[Layer] = model.layers
    for l in layers:
        if l.__class__ == LayerTypes.DENSE.value:
            layer_dicts.extend(_save_dense(l))

    model_dict = {
        'name': model.name,
        'layers': model.layers
    }


def main():
    model = load_model('model.h5')
    save_model(model)


if __name__ == '__main__':
    main()
