import argparse
import time
import os
import random
from PIL import Image
from detector import DefectDetector
from analysis import DefectAnalyzer
from data_loader import WM811KDataset


def image_generator(dataset_path, total_count, batch_size, seed=42):
    """Generates batches of real images from the dataset."""
    dataset = WM811KDataset(dataset_path, transform=None)

    all_samples = dataset.samples
    random.Random(seed).shuffle(all_samples)

    samples_to_process = all_samples[:total_count]

    current_idx = 0
    while current_idx < len(samples_to_process):
        batch_paths = samples_to_process[current_idx: current_idx + batch_size]
        batch_images = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Error reading {p}: {e}")

        if batch_images:
            yield batch_images

        current_idx += len(batch_paths)


def main():
    parser = argparse.ArgumentParser(description="High-Speed Edge-AI Defect Detection Demo")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to an image-folder-style dataset (root/<class>/*.png)")
    parser.add_argument("--model_path", type=str, default="best_sem_model.pth",
                         help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="sem", choices=["sem", "wm811k"],
                         help="Which class list / no-defect label to use -- must match model_path")
    parser.add_argument("--count", type=int, default=100, help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Dataset not found at {args.data_dir}. Please run download_dataset.py first, "
              f"or pass --data_dir pointing at your local copy.")
        return

    detector = DefectDetector(use_cuda=args.cuda, model_path=args.model_path, dataset=args.dataset)
    analyzer = DefectAnalyzer()

    print(f"Starting processing of {args.count} images with batch size {args.batch_size} (Real Data)...")

    start_total = time.perf_counter()

    processed_count = 0

    for batch in image_generator(args.data_dir, args.count, args.batch_size):
        results = detector.detect_batch(batch)
        analyzer.analyze_batch_results(results)

        processed_count += len(batch)
        if processed_count % 500 == 0:
            print(f"Processed {processed_count}/{args.count}...")

    end_total = time.perf_counter()
    total_time = end_total - start_total

    stats = analyzer.get_global_stats()
    avg_latency = total_time / processed_count if processed_count > 0 else 0

    print("\n" + "=" * 40)
    print("       PERFORMANCE RESULTS       ")
    print("=" * 40)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Time:   {total_time:.4f} sec")
    print(f"Throughput:   {stats['total_images'] / total_time:.2f} img/sec" if total_time > 0 else "Throughput: N/A")
    print(f"Avg Latency:  {avg_latency * 1000:.4f} ms/image")
    print("-" * 40)
    print(f"Defects Found: {stats['total_defects']}")
    print(f"Defect Distribution: {stats['defect_distribution']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
