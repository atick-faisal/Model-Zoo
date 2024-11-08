import argparse
import os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from ultralytics.utils.torch_utils import model_info

from dataloader import create_dataloaders


class PSNRLoss(nn.Module):
    def __init__(self, max_val: float = 1.0) -> None:
        """
        Args:
            max_val: Maximum value of the signal (1.0 for normalized images)
        """
        super().__init__()
        self.max_val = torch.tensor(max_val)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the negative PSNR (we minimize this value)
        Args:
            output: Predicted images
            target: Ground truth images
        Returns:
            Negative PSNR value (scalar tensor)
        """
        # Ensure max_val is on the same device as the inputs
        self.max_val = self.max_val.to(output.device)

        # Calculate MSE per image
        mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])

        # Handle cases where mse is 0
        zero_mask = (mse == 0)
        mse = torch.where(zero_mask, torch.tensor(1e-8).to(mse.device), mse)

        # Calculate PSNR
        psnr = 20 * torch.log10(self.max_val) - 10 * torch.log10(mse)

        # Set PSNR to a large value where mse was 0
        psnr = torch.where(zero_mask, torch.tensor(100.0).to(psnr.device), psnr)

        return -torch.mean(psnr)  # Negative because we want to maximize PSNR


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class UNetRegressor(nn.Module):
    def __init__(self, init_features: int = 64) -> None:
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(3, init_features)
        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.enc4 = DoubleConv(init_features * 4, init_features * 8)
        self.enc5 = DoubleConv(init_features * 8, init_features * 16)

        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2,
                                      stride=2)
        self.dec4 = DoubleConv(init_features * 16, init_features * 8)

        self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(init_features * 8, init_features * 4)

        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(init_features * 4, init_features * 2)

        self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)

        self.final_conv = nn.Conv2d(init_features, 3, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        # Bridge
        enc5 = self.enc5(pool4)

        # Decoder
        up4 = self.up4(enc5)
        concat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(concat4)

        up3 = self.up3(dec4)
        concat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(concat1)

        return self.final_conv(dec1)


class FusionGenerator(nn.Module):
    def __init__(self):
        super(FusionGenerator, self).__init__()

        # First layer: 5x5 conv with padding=2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Second layer: 5x5 conv with padding=2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Third layer: 3x3 conv with padding=1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Fourth layer: 3x3 conv with padding=1
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Fifth layer: 1x1 conv with tanh activation
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0),
            # nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the generator
        Args:
            x: Input tensor of shape (batch_size, 2, H, W) containing concatenated IR and visible images
        Returns:
            Fused image tensor of shape (batch_size, 1, H, W)
        """
        x = self.conv1(x)  # Output: (batch_size, 64, H, W)
        x = self.conv2(x)  # Output: (batch_size, 128, H, W)
        x = self.conv3(x)  # Output: (batch_size, 256, H, W)
        x = self.conv4(x)  # Output: (batch_size, 128, H, W)
        x = self.conv5(x)  # Output: (batch_size, 1, H, W)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection handling
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        return self.res_block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv_transpose(x)))
        x = self.res_block(x)
        return x


class ImageGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_filters=64):
        super(ImageGenerator, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU()
        )

        # Encoder pathway
        self.encoder1 = EncoderBlock(base_filters, base_filters * 2)
        self.encoder2 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.encoder3 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bridge
        self.bridge = ResidualBlock(base_filters * 8, base_filters * 8)

        # Decoder pathway
        self.decoder3 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.decoder2 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.decoder1 = DecoderBlock(base_filters * 2, base_filters)

        # Final convolution
        self.final = nn.Conv2d(base_filters, output_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # Encoder
        x1 = self.initial(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # Bridge
        bridge = self.bridge(x4)

        # Decoder with skip connections
        d3 = self.decoder3(bridge) + x3
        d2 = self.decoder2(d3) + x2
        d1 = self.decoder1(d2) + x1

        # Final output
        output = torch.tanh(self.final(d1))

        return output


def init_weights(model):
    """Initialize network weights using Xavier initialization"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


# Example usage
def create_model(input_channels=3, output_channels=3, base_filters=64):
    model = ImageGenerator(input_channels, output_channels, base_filters)
    init_weights(model)
    return model

def compute_losses(
        output: Tensor,
        target: Tensor,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        mse_weight: float,
        psnr_weight: float
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute combined loss and individual metrics
    """
    mse_loss = mse_criterion(output, target)
    psnr_loss = psnr_criterion(output, target)

    # Combine losses
    total_loss = mse_weight * mse_loss + psnr_weight * psnr_loss

    # Store individual loss values
    loss_dict = {
        'mse_loss': mse_loss.item(),
        'psnr_loss': psnr_loss.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, loss_dict


def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace
) -> Dict[str, float]:
    model.train()
    total_losses = {'mse_loss': 0.0, 'psnr_loss': 0.0, 'total_loss': 0.0}
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (input_images, target_images) in enumerate(pbar):
        input_images, target_images = input_images.to(device), target_images.to(device)

        optimizer.zero_grad()
        output = model(input_images)

        loss, loss_dict = compute_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        loss.backward()
        optimizer.step()

        # Update running losses
        for k, v in loss_dict.items():
            total_losses[k] += v

        if batch_idx % args.log_interval == 0:
            wandb.log({
                'batch_total_loss': loss_dict['total_loss'],
                'batch_mse_loss': loss_dict['mse_loss'],
                'batch_psnr_loss': loss_dict['psnr_loss'],
                'batch': batch_idx + epoch * len(train_loader)
            })

        pbar.set_postfix(loss_dict)

    # Compute averages
    avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
        model: nn.Module,
        val_loader: DataLoader,
        mse_criterion: nn.Module,
        psnr_criterion: nn.Module,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace
) -> Dict[str, float]:
    model.eval()
    total_losses = {'mse_loss': 0.0, 'psnr_loss': 0.0, 'total_loss': 0.0}
    vis_images: List[Tensor] = []

    for batch_idx, (input_images, target_images) in enumerate(val_loader):
        input_images, target_images = input_images.to(device), target_images.to(device)
        output = model(input_images)

        _, loss_dict = compute_losses(
            output, target_images,
            mse_criterion, psnr_criterion,
            args.mse_weight, args.psnr_weight
        )

        # Update running losses
        for k, v in loss_dict.items():
            total_losses[k] += v

        if batch_idx == 0:
            vis_images = [
                input_images[:args.num_samples].cpu(),
                output[:args.num_samples].cpu(),
                target_images[:args.num_samples].cpu()
            ]

    # Compute averages
    avg_losses = {k: v / len(val_loader) for k, v in total_losses.items()}

    if epoch % args.vis_interval == 0:
        vis_grid = make_grid(torch.cat(vis_images, dim=0), nrow=args.num_samples)
        wandb.log({
            "examples": wandb.Image(
                vis_grid,
                caption=f"Top: Input, Middle: Output, Bottom: Target (Epoch {epoch})"
            )
        })

    return avg_losses


def main() -> None:
    parser = argparse.ArgumentParser(description='UNet Regressor Training')

    # [Previous arguments remain the same]
    parser.add_argument('--data-root', type=str, required=True, help='root directory for data')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--init-features', type=int, default=64, help='initial number of features in UNet')

    # Loss weights
    parser.add_argument('--mse-weight', type=float, default=1.0, help='weight for MSE loss')
    parser.add_argument('--psnr-weight', type=float, default=0.1, help='weight for PSNR loss')

    # [Rest of the arguments remain the same]
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=10, help='log metrics every N batches')
    parser.add_argument('--vis-interval', type=int, default=5, help='visualize examples every N epochs')
    parser.add_argument('--num-samples', type=int, default=4, help='number of examples to visualize')
    parser.add_argument('--wandb-project', type=str, default='unet-regressor', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # model = UNetRegressor(init_features=args.init_features).to(device)
    # model = FusionGenerator().to(device)
    model = create_model()
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()
    psnr_criterion = PSNRLoss()

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    best_val_loss: float = float('inf')
    for epoch in range(args.epochs):
        train_losses = train_epoch(
            model, train_loader, mse_criterion, psnr_criterion,
            optimizer, device, epoch, args
        )
        val_losses = validate(
            model, val_loader, mse_criterion, psnr_criterion,
            device, epoch, args
        )

        # Log all losses
        wandb.log({
            'epoch': epoch,
            'train_total_loss': train_losses['total_loss'],
            'train_mse_loss': train_losses['mse_loss'],
            'train_psnr_loss': train_losses['psnr_loss'],
            'val_total_loss': val_losses['total_loss'],
            'val_mse_loss': val_losses['mse_loss'],
            'val_psnr_loss': val_losses['psnr_loss']
        })

        # Save best model based on total validation loss
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)

    wandb.finish()


if __name__ == "__main__":
    main()
