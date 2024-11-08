import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from torch import Tensor


class ImagePairDataset(Dataset):
    def __init__(
            self,
            root_dir: str | Path,
            split: str = 'train',
            transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            root_dir: Root directory containing train and val folders
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
        """
        self.root_dir: Path = Path(root_dir)
        self.split: str = split
        self.transform: Callable = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Get paths for input and target directories
        self.input_dir: Path = self.root_dir / split / 'input'
        self.target_dir: Path = self.root_dir / split / 'target'

        # Get list of image files
        self.image_files: List[str] = sorted([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Verify that all input images have corresponding target images
        for img_file in self.image_files:
            assert (self.target_dir / img_file).exists(), \
                f"Target image not found for {img_file}"

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name: str = self.image_files[idx]

        # Read input and target images
        input_path: Path = self.input_dir / img_name
        target_path: Path = self.target_dir / img_name

        input_image: Image.Image = Image.open(input_path).convert('RGB')
        target_image: Image.Image = Image.open(target_path).convert('RGB')

        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def create_dataloaders(
        data_root: str | Path,
        batch_size: int,
        num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from the given data root directory.

    Args:
        data_root: Path to root directory containing train and val folders
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform: Compose = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset: ImagePairDataset = ImagePairDataset(
        root_dir=data_root,
        split='train',
        transform=transform
    )

    val_dataset: ImagePairDataset = ImagePairDataset(
        root_dir=data_root,
        split='val',
        transform=transform
    )

    # Create dataloaders
    train_loader: DataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader: DataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# # Example usage:
# if __name__ == "__main__":
#     data_root: str = "path/to/your/data"
#     train_loader, val_loader = create_dataloaders(
#         data_root=data_root,
#         batch_size=32,
#         num_workers=4
#     )
#
#     # Print dataset sizes
#     print(f"Training samples: {len(train_loader.dataset)}")
#     print(f"Validation samples: {len(val_loader.dataset)}")
#
#     # Test loading a batch
#     for inputs, targets in train_loader:
#         print(f"Input batch shape: {inputs.shape}")
#         print(f"Target batch shape: {targets.shape}")
#         break