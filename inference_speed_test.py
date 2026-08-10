import argparse
import torch
import time
from model import SimpleCNN
from PIL import Image
from torchvision import transforms
import os


def run_inference_test(model_path, num_classes, image_path=None, num_runs=100):
    device = torch.device("cpu")  # CPU-only benchmark: this is the deployment target.
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    else:
        print("Using dummy input (random noise).")
        input_tensor = torch.randn(1, 3, 96, 96).to(device)

    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    print(f"Running benchmark for {num_runs} iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.perf_counter()

    avg_latency = (end_time - start_time) / num_runs * 1000  # ms
    throughput = num_runs / (end_time - start_time)  # img/s

    print("\nResults:")
    print(f"Device: {device}")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")

    if avg_latency < 10:
        print("\n[SUCCESS] Latency is under 10 ms!")
    else:
        print("\n[WARNING] Latency is over 10 ms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency for a trained checkpoint.")
    parser.add_argument("--model_path", type=str, default="best_sem_model.pth")
    parser.add_argument("--num_classes", type=int, default=8,
                         help="Must match the checkpoint (8 for SEM, 9 for WM811K)")
    parser.add_argument("--image_path", type=str, default=None,
                         help="Optional real image to test with; otherwise uses random noise")
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path}")

    run_inference_test(args.model_path, args.num_classes, args.image_path, args.num_runs)
