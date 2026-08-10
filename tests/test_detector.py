"""
Basic smoke tests. Run with: pytest tests/

The most important one here is test_no_defect_labels_exist_in_class_lists --
this is exactly the check that would have caught the original bug, where
detector.py checked for a 'none' class that didn't exist in the SEM class
list (the real label was 'Clean'), silently marking every prediction as
a defect.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import DefectDetector, WM811K_CLASSES, SEM_CLASSES, NO_DEFECT_LABELS
from model import SimpleCNN


def test_no_defect_labels_exist_in_class_lists():
    """Each dataset's no-defect label(s) must actually be present in its class list."""
    for dataset, classes in [("wm811k", WM811K_CLASSES), ("sem", SEM_CLASSES)]:
        missing = NO_DEFECT_LABELS[dataset] - set(classes)
        assert not missing, (
            f"NO_DEFECT_LABELS['{dataset}'] contains label(s) {missing} "
            f"not present in the class list {classes}."
        )


def test_model_output_shape():
    """Model should produce logits of shape (batch, num_classes) for a 96x96 RGB input."""
    for num_classes in (8, 9):
        model = SimpleCNN(num_classes=num_classes)
        model.eval()
        dummy = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, num_classes)


def test_detect_batch_runs_without_a_real_checkpoint_raises():
    """DefectDetector should refuse to run with a missing checkpoint rather than
    silently falling back to random weights (which would produce meaningless
    predictions that look like real results)."""
    try:
        DefectDetector(model_path="this_file_does_not_exist.pth", dataset="sem")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for a missing checkpoint.")


def test_bad_dataset_no_defect_mismatch_is_rejected():
    """If someone edits NO_DEFECT_LABELS/classes and introduces a mismatch again,
    construction should fail loudly instead of silently mislabeling every image."""
    import detector as detector_module
    original = dict(detector_module.NO_DEFECT_LABELS)
    try:
        detector_module.NO_DEFECT_LABELS["sem"] = {"none"}  # wrong on purpose
        raised = False
        try:
            DefectDetector(model_path="best_sem_model.pth", dataset="sem")
        except ValueError:
            raised = True
        assert raised, "Expected a ValueError when no-defect label isn't in the class list."
    finally:
        detector_module.NO_DEFECT_LABELS.clear()
        detector_module.NO_DEFECT_LABELS.update(original)


if __name__ == "__main__":
    test_no_defect_labels_exist_in_class_lists()
    test_model_output_shape()
    test_detect_batch_runs_without_a_real_checkpoint_raises()
    test_bad_dataset_no_defect_mismatch_is_rejected()
    print("All smoke tests passed.")
