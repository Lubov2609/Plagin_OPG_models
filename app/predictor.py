from pathlib import Path

import torch


class OPGPredictor:
    def __init__(self, model_path, model_class, preprocessor, threshold=0.5, local_model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self.local_model_path = Path(local_model_path) if local_model_path else None
        self.model_class = model_class
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.model = None
        self.active_model_path = None
        self.model_source = "base"

        self.wisdom_teeth_classes = {
            0: "8-е зубы отсутствуют",
            1: "Обнаружен 1 восьмой зуб",
            2: "Обнаружены 2 восьмых зуба",
            3: "Обнаружены 3 восьмых зуба",
            4: "Обнаружены 4 восьмых зуба",
        }

        self.impacted_wisdom_teeth_classes = {
            0: "Ретинированные 8-е зубы отсутствуют",
            1: "Обнаружен 1 ретинированный 8-й зуб",
            2: "Обнаружены 2 ретинированных 8-х зуба",
            3: "Обнаружены 3 ретинированных 8-х зуба",
            4: "Обнаружены 4 ретинированных 8-х зуба",
        }

        self.reload_model()

    def _candidate_model_paths(self):
        paths = []
        if self.local_model_path and self.local_model_path.exists():
            paths.append((self.local_model_path, "local"))
        paths.append((self.model_path, "base"))
        return paths

    def _load_state_dict(self, model, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)

    def reload_model(self):
        last_error = None

        for model_path, source in self._candidate_model_paths():
            try:
                model = self.model_class()
                self._load_state_dict(model, model_path)
                model.to(self.device)
                model.eval()

                self.model = model
                self.active_model_path = model_path
                self.model_source = source
                return {
                    "source": source,
                    "path": str(model_path),
                }
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Не удалось загрузить модель: {last_error}")

    def get_model_status(self):
        if self.model_source == "local":
            return "Активна локально дообученная модель"
        return "Активна исходная модель"

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
                "confidence_level": self._confidence_level(caries_prob),
            },
            "pulpitis": {
                "title": "Пульпит",
                "detected": bool(pulpitis_prob >= self.threshold),
                "status": "Обнаружен" if pulpitis_prob >= self.threshold else "Не обнаружен",
                "confidence": round(pulpitis_prob, 4),
                "confidence_percent": round(pulpitis_prob * 100, 2),
                "confidence_level": self._confidence_level(pulpitis_prob),
            },
            "wisdom_teeth": {
                "title": "Наличие 8-х зубов",
                "class_id": wisdom_class,
                "class_name": self.wisdom_teeth_classes[wisdom_class],
                "confidence": round(wisdom_conf, 4),
                "confidence_percent": round(wisdom_conf * 100, 2),
                "confidence_level": self._confidence_level(wisdom_conf),
                "probabilities": [round(float(p), 4) for p in wisdom_probs],
            },
            "impacted_wisdom_teeth": {
                "title": "Ретинированные 8-е зубы",
                "class_id": impacted_class,
                "class_name": self.impacted_wisdom_teeth_classes[impacted_class],
                "confidence": round(impacted_conf, 4),
                "confidence_percent": round(impacted_conf * 100, 2),
                "confidence_level": self._confidence_level(impacted_conf),
                "probabilities": [round(float(p), 4) for p in impacted_probs],
            },
        }
