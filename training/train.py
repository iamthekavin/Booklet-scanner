"""
YOLO11 Training Script for VEE Scanner.

Usage:
  python training/train.py train --model yolo11s.pt --epochs 100 --batch 16
  python training/train.py train --model yolo11s-obb.pt --epochs 150 --obb
"""

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    model = YOLO(args.model)
    
    # Optional OBB specific configs can go here based on args.obb
    
    results = model.train(
        data=str(Path(__file__).parent / 'dataset.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/detect',
        name=args.name,
        exist_ok=True,
        patience=20,  # early stopping
        # Augmentations tuned for scanning scenario
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15.0,  # booklet might be slightly rotated
        translate=0.1,
        scale=0.3,
        flipud=0.0,  # don't flip upside down (text orientation matters later)
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
    )
    
    # Export best model to ONNX
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        best_model.export(format='onnx', dynamic=True, simplify=True)
        
        print('Training complete!')
        print(f'Best model: {best_weights}')
        print(f'ONNX export: {best_weights.with_suffix(".onnx")}')
    else:
        print('Training completed, but best weights not found.')


def validate(args: argparse.Namespace) -> None:
    """Run the validation pipeline."""
    model = YOLO(args.model)
    results = model.val(
        data=str(Path(__file__).parent / 'dataset.yaml'),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    if hasattr(results, 'box'):
        print(f'mAP50: {results.box.map50:.4f}')
        print(f'mAP50-95: {results.box.map:.4f}')
    elif hasattr(results, 'obb'):
        print(f'mAP50: {results.obb.map50:.4f}')
        print(f'mAP50-95: {results.obb.map:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description="VEE Scanner YOLO Training Script")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, default='yolo11s.pt', help='Base model weights')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--device', type=str, default='', help='Device id (e.g. 0 or cpu)')
    train_parser.add_argument('--name', type=str, default='vee_booklet', help='Experiment name')
    train_parser.add_argument('--obb', action='store_true', help='Use Oriented Bounding Boxes')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate the model')
    val_parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    val_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    val_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    val_parser.add_argument('--device', type=str, default='', help='Device id')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'validate':
        validate(args)

if __name__ == '__main__':
    main()
