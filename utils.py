import pickle
import torch
import inspect


def save_model(model, params, directory):
    with open(f'{directory}/params.pkl', 'wb') as f:
        pickle.dump(params, f)
    with open(f'{directory}/model.pt', 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model_class, directory):
    with open(f'{directory}/params.pkl', 'rb') as f:
        params = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = list(inspect.signature(model_class.__init__).parameters.keys())
    params_class = dict()
    for a in args[1:]:
        params_class[a] = params[a]
    model = model_class(**params_class).to(device)
    with open(f'{directory}/model.pt', 'rb') as f:
        state_dict = torch.load(f, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, params
