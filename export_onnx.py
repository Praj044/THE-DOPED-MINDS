import argparse
import torch
import torch.onnx
from model import SimpleCNN
import os
import onnx
import onnxruntime as ort
import numpy as np


def export_model(model_path, output_path, num_classes):
    print(f"Loading model from {model_path}...")

    model = SimpleCNN(num_classes=num_classes)

    if not os.path.exists(model_path):
        raise SystemExit(f"Error: Model file {model_path} not found.")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    dummy_input = torch.randn(1, 3, 96, 96, requires_grad=True)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(model,
                       dummy_input,
                       output_path,
                       export_params=True,
                       opset_version=11,
                       do_constant_folding=True,
                       input_names=['input'],
                       output_names=['output'],
                       dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})

    print("Export complete.")

    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    print("Testing inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    torch_out = model(dummy_input)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the WM811K checkpoint to ONNX.")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--output_path", type=str, default="model.onnx")
    parser.add_argument("--num_classes", type=int, default=9,
                         help="9 for the WM811K classes "
                              "['Center','Donut','Edge Local','Edge Ring','Local','Scratch','near full','none','random']")
    args = parser.parse_args()

    export_model(args.model_path, args.output_path, args.num_classes)
