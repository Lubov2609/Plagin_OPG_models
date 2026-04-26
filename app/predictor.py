import torch


class OPGPredictor:
    def __init__(self, model_path, model_class, preprocessor, threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model_class()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = preprocessor
        self.threshold = threshold

        self.wisdom_teeth_classes = {
            0: "8-е зубы отсутствуют",
            1: "Обнаружен 1 восьмой зуб",
            2: "Обнаружены 2 восьмых зуба",
            3: "Обнаружены 3 восьмых зуба",
            4: "Обнаружены 4 восьмых зуба"
        }

        self.impacted_wisdom_teeth_classes = {
            0: "Ретенированные 8-е зубы отсутствуют",
            1: "Обнаружен 1 ретенированный 8-й зуб",
            2: "Обнаружены 2 ретенированных 8-х зуба",
            3: "Обнаружены 3 ретенированных 8-х зуба",
            4: "Обнаружены 4 ретенированных 8-х зуба"
        }

    def _confidence_level(self, value):
        if value >= 0.75:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"

    def predict(self, image_path):
        image_tensor = self.preprocessor.preprocess(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        caries_prob = torch.sigmoid(outputs[0]).item()
        pulpitis_prob = torch.sigmoid(outputs[1]).item()

        wisdom_probs = torch.softmax(outputs[2], dim=1)[0]
        impacted_probs = torch.softmax(outputs[3], dim=1)[0]

        wisdom_class = int(torch.argmax(wisdom_probs).item())
        impacted_class = int(torch.argmax(impacted_probs).item())

        wisdom_conf = float(wisdom_probs[wisdom_class])
        impacted_conf = float(impacted_probs[impacted_class])

        return {
            "caries": {
                "title": "Кариес",
                "detected": bool(caries_prob >= self.threshold),
                "status": "Обнаружен" if caries_prob >= self.threshold else "Не обнаружен",
                "confidence": round(caries_prob, 4),
                "confidence_percent": round(caries_prob * 100, 2),
                "confidence_level": self._confidence_level(caries_prob)
            },
            "pulpitis": {
                "title": "Пульпит",
                "detected": bool(pulpitis_prob >= self.threshold),
                "status": "Обнаружен" if pulpitis_prob >= self.threshold else "Не обнаружен",
                "confidence": round(pulpitis_prob, 4),
                "confidence_percent": round(pulpitis_prob * 100, 2),
                "confidence_level": self._confidence_level(pulpitis_prob)
            },
            "wisdom_teeth": {
                "title": "Наличие 8-х зубов",
                "class_id": wisdom_class,
                "class_name": self.wisdom_teeth_classes[wisdom_class],
                "confidence": round(wisdom_conf, 4),
                "confidence_percent": round(wisdom_conf * 100, 2),
                "confidence_level": self._confidence_level(wisdom_conf),
                "probabilities": [
                    round(float(p), 4) for p in wisdom_probs
                ]
            },
            "impacted_wisdom_teeth": {
                "title": "Ретенированные 8-е зубы",
                "class_id": impacted_class,
                "class_name": self.impacted_wisdom_teeth_classes[impacted_class],
                "confidence": round(impacted_conf, 4),
                "confidence_percent": round(impacted_conf * 100, 2),
                "confidence_level": self._confidence_level(impacted_conf),
                "probabilities": [
                    round(float(p), 4) for p in impacted_probs
                ]
            }
        }