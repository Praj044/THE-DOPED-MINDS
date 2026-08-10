import torch
import time
import numpy as np
from PIL import Image
from model import SimpleCNN
import torchvision.transforms as transforms

# Class lists for the two trained models. Keep these in sync with the
# order produced by data_loader.WM811KDataset (alphabetically sorted
# subfolder names) at the time each model was trained.
WM811K_CLASSES = ['Center', 'Donut', 'Edge Local', 'Edge Ring', 'Local',
                   'Scratch', 'near full', 'none', 'random']
SEM_CLASSES = ['Bridge', 'CMP scratch', 'Clean', 'LER', 'crack',
               'manforsed via', 'open', 'short']

# Which label(s) in each class list mean "no defect". This must match the
# actual class the dataset uses -- getting this wrong silently flips every
# prediction (this previously happened here: the SEM classes use 'Clean',
# not 'none', but has_defect was hardcoded to check for 'none').
NO_DEFECT_LABELS = {
    "wm811k": {"none"},
    "sem": {"Clean"},
}


class DefectDetector:
    def __init__(self, use_cuda=False, model_path="best_sem_model.pth",
                 dataset="sem"):
        """
        Args:
            use_cuda: run on GPU if available.
            model_path: path to a .pth checkpoint saved by train.py / train_sem.py.
            dataset: "sem" or "wm811k" -- selects the class list and the
                     label(s) considered "no defect". Must match the dataset
                     the checkpoint at model_path was actually trained on.
        """
        if dataset not in NO_DEFECT_LABELS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Expected one of {list(NO_DEFECT_LABELS)}."
            )

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Initializing DefectDetector on {self.device}...")

        self.dataset = dataset
        self.classes = SEM_CLASSES if dataset == "sem" else WM811K_CLASSES
        self.no_defect_labels = NO_DEFECT_LABELS[dataset]

        # Sanity check: the "no defect" label(s) must actually exist in the
        # class list, otherwise has_defect() would be True for every
        # prediction (which is the exact bug this class previously had).
        missing = self.no_defect_labels - set(self.classes)
        if missing:
            raise ValueError(
                f"No-defect label(s) {missing} not found in classes {self.classes}. "
                "Fix NO_DEFECT_LABELS / the classes list before running inference."
            )

        num_classes = len(self.classes)
        self.model = SimpleCNN(num_classes=num_classes)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file {model_path} not found. Refusing to run inference with "
                "randomly-initialized weights -- pass a real checkpoint path."
            )

        self.model.to(self.device)
        self.model.eval()

        # Transform must match training (resize to 96x96)
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dummy warmup
        dummy_input = torch.randn(1, 3, 96, 96).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)
        print("Model initialized and warmed up.")

    def preprocess(self, images):
        """
        Convert list of numpy images (or PIL images) to tensor batch.
        """
        batch_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            batch_tensors.append(self.transform(img))

        return torch.stack(batch_tensors).to(self.device)

    def detect_batch(self, images):
        """
        Run detection on a batch of images.
        """
        if not images:
            return []

        start_time = time.perf_counter()

        input_tensor = self.preprocess(images)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

        inference_time = (time.perf_counter() - start_time)
        per_image_time = inference_time / len(images)

        results = []
        cpu_preds = predictions.cpu().numpy()
        cpu_probs = probs.cpu().numpy()

        for i, pred_idx in enumerate(cpu_preds):
            pred_class = self.classes[pred_idx]
            score = float(cpu_probs[i][pred_idx])

            has_defect = pred_class not in self.no_defect_labels

            results.append({
                "image_idx": i,
                "has_defect": has_defect,
                "confidence": score,
                "latency_sec": per_image_time,
                "defect_type": pred_class
            })

        return results
