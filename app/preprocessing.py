from PIL import Image
from torchvision import transforms


class OPGPreprocessor:
    def __init__(self):
        self.size = (448, 224)

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("L")
        image = image.resize(self.size)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor