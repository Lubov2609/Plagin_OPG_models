import torch


class ModelLoader:
    def __init__(self, model_class, model_path: str, device: str = None):
        self.model_class = model_class
        self.model_path = model_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def load_model(self):
        model = self.model_class()

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model, self.device