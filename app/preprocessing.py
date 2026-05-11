from PIL import Image
from torchvision import transforms


class OPGPreprocessor:
    def __init__(self):
        # PIL size is (width, height); the training project uses H=448, W=896.
        self.size = (896, 448)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("L")
        image = image.resize(self.size)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor
