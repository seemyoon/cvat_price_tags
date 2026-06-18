import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class ModelHandler:
    def __init__(self, labels):
        self.labels = labels
        model_path = hf_hub_download(
            repo_id="openfoodfacts/price-tag-detection",
            filename="weights/best.pt",
        )
        self.model = YOLO(model_path)

    def infer(self, image, threshold):
        results = self.model.predict(
            source=np.array(image),
            conf=threshold,
            verbose=False,
        )

        detections = []
        h, w = np.array(image).shape[:2]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "confidence": str(score),
                    "label": self.labels.get(cls_id, "price_tag"),
                    "points": [
                        max(int(x1), 0),
                        max(int(y1), 0),
                        min(int(x2), w),
                        min(int(y2), h),
                    ],
                    "type": "rectangle",
                })

        return detections
