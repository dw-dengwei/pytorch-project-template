from .toy_model import ToyModel


models = {
    'ToyModel': ToyModel
}

def  get_model(model_name, *args, **kwargs):
    return models[model_name](*args, **kwargs)