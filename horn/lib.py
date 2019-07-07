from keras.models import load_model
from model_saver import ModelSaver


def main():
    ms = ModelSaver()
    model = load_model('model.h5')
    ms.save_model(model, 'model.horn')


if __name__ == '__main__':
    main()
