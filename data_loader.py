import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob


class WM811KDataset(Dataset):
    """
    Generic image-folder-style dataset: expects root_dir/<class_name>/*.{jpg,png,...}
    Used for both the WM811K wafer-map dataset and the SEM defect dataset --
    the class list is simply whatever subfolders exist under root_dir.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.samples.extend(glob.glob(os.path.join(class_dir, ext)))
                self.samples.extend(glob.glob(os.path.join(class_dir, ext.upper())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


class _TransformSubset(Dataset):
    """Applies its own transform to a fixed list of underlying indices,
    independent of any other view over the same base dataset."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img_path = self.base_dataset.samples[self.indices[i]]
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.base_dataset.class_to_idx[class_name]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=0, seed=42):
    """
    Splits data_dir into train/val DataLoaders. Uses a fixed seed by default
    so the split is reproducible. Train and val get their own transform
    (augmentation only on train) via independent dataset views, so setting
    one doesn't silently affect the other.
    """
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform=None here: _TransformSubset does its own image loading/transform.
    base_dataset = WM811KDataset(data_dir, transform=None)

    n = len(base_dataset)
    val_size = int(n * val_split)
    train_size = n - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_dataset = _TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset = _TransformSubset(base_dataset, val_indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, base_dataset.classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the dataloader against a real dataset folder.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to an image-folder-style dataset (root/<class>/*.jpg)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Path not found: {args.data_dir}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
