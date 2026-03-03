import os
import numpy as np
import torch
import json
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

from src.data.download import ESC50Downloader
from src.data.augment import AudioAugment
from src.models.cnn import CNN
from src.models.transformer import AST
from src.models.predict import AudioPredictor
from src.models.traincnn import CNNTrainer
from src.models.traintransformer import TransformerTrainer
from src.config.config import ProcessingConfig, DatasetConfig, DownloadConfig, TrainConfig

def _load_or_preprocess(args) -> tuple[np.ndarray, np.ndarray]:
    X_path = args.X_path or "data/preprocessed/X.npy"
    y_path = args.y_path or "data/preprocessed/y.npy"
    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading existing processed data...")
        return np.load(X_path, allow_pickle=True), np.load(y_path)
    print("Processing audio data...")
    augmenter = AudioAugment()
    augmenter.run(augment=True, preprocess=True)
    return np.load(X_path, allow_pickle=True), np.load(y_path)

def _build_config(args, **overrides) -> TrainConfig:
    return TrainConfig(**{k: v for k, v in overrides.items() if v is not None})

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESC50 Audio Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    download_parser = subparsers.add_parser('download', help='Download ESC50 dataset')
    download_parser.set_defaults(func=cmd_download)

    augment_parser = subparsers.add_parser('augment', help='Create augmented datasets')
    augment_parser.add_argument('--input-dir', type=str, default="data/audio/0", help="Input audio directory")
    augment_parser.add_argument('--output-dir', type=str, default="data/audio", help='Output directory for augmented datasets')
    augment_parser.set_defaults(func=cmd_augment)

    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess audio dataset')
    preprocess_parser.add_argument('--input-dir', type=str, default="data/audio", help='Input audio directory')
    preprocess_parser.add_argument('--output-dir', type=str, default="data/preprocessed", help='Output directory for preprocessed data')
    preprocess_parser.set_defaults(func=cmd_preprocess)

    train_parser = subparsers.add_parser('train_cnn', help='Train CNN model')
    train_parser.add_argument('--audio-dir', type=str, help='Path to training audio directory')
    train_parser.add_argument('--output-dir', type=str, help='Path to save preprocessed data')
    train_parser.add_argument('--X-path', type=str, help='Path to preprocessed X.npy')
    train_parser.add_argument('--y-path', type=str, help='Path to preprocessed y.npy')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (default: 100)')
    train_parser.add_argument('--lr', type=float, help='Learning rate (default: 0.01)')
    train_parser.add_argument('--sample-fraction', type=float, help='Fraction of samples per epoch (default: 1/8)')
    train_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    train_parser.add_argument('--save-every', type=int, help='Save checkpoint every N epochs')
    train_parser.set_defaults(func=cmd_train_cnn)

    train_parser = subparsers.add_parser('train_transformer', help='Train Transformer model')
    train_parser.add_argument('--audio-dir', type=str, help='Path to training audio directory')
    train_parser.add_argument('--output-dir', type=str, help='Path to save preprocessed data')
    train_parser.add_argument('--X-path', type=str, help='Path to preprocessed X.npy')
    train_parser.add_argument('--y-path', type=str, help='Path to preprocessed y.npy')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (default: 100)')
    train_parser.add_argument('--lr', type=float, help='Learning rate (default: 0.01)')
    train_parser.add_argument('--sample-fraction', type=float, help='Fraction of samples per epoch (default: 1/8)')
    train_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    train_parser.add_argument('--save-every', type=int, help='Save checkpoint every N epochs')
    train_parser.set_defaults(func=cmd_train_transformer)

    # resume_parser = subparsers.add_parser('resume', help='Resume training from checkpoint for CNN models')
    # resume_parser.add_argument('--resume-from', type=str, required=True, help='Path to checkpoint file')
    # resume_parser.add_argument('--X-path', type=str, default="data/preprocessed/X.npy")
    # resume_parser.add_argument('--y-path', type=str, default="data/preprocessed/y.npy")
    # resume_parser.add_argument('--epochs', type=int, help='Number of epochs (default: 100)')
    # resume_parser.add_argument('--batch-size', type=int, help='Batch size (default: 100)')
    # resume_parser.add_argument('--lr', type=float, help='Learning rate (default: 0.01)')
    # resume_parser.add_argument('--sample-fraction', type=float, help='Fraction of samples per epoch')
    # resume_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    # resume_parser.add_argument('--save-every', type=int, help='Save checkpoint every N epochs')
    # resume_parser.set_defaults(func=cmd_resume_cnn)

    predict_parser = subparsers.add_parser('predict', help='Predict audio file class')
    predict_parser.add_argument('audio_file', type=str, help='Path to .wav file to classify')
    predict_parser.add_argument('--model', type=str, default='final_model.pt', help='Path to model checkpoint (default: best_model.pt)')
    predict_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions (default: 5)')
    predict_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: auto)')
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)

def cmd_download(args) -> None:
    print("Downloading ESC50 audio data...")

    downloader = ESC50Downloader()

    downloader.download_clean()

    print("Downloaded and cleaned data.")

def cmd_augment(args) -> None:
    print("Augmenting audio data...")

    augmentater = AudioAugment()

    augmentater.run(augment=True, preprocess=False)

    print(f"Augmented data and saved to {args.output_dir}")

def cmd_preprocess(args) -> None:
    print("Processing audio data...")
    
    augmentater = AudioAugment()

    augmentater.run(augment=False, preprocess=True)

    print(f"Preprocessed data and saved to {args.output_dir}")

def cmd_train_cnn(args) -> None: 
    X, y = _load_or_preprocess(args)
    trainer = CNNTrainer(_build_config(args,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_epoch_fraction=args.sample_fraction,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epoch=args.save_every,
    ))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_val_acc = trainer.train_cnn(CNN(n_classes=len(np.unique(y))), X_train, y_train, X_val, y_val, fold_num=0)
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

def cmd_train_transformer(args) -> None:
    X, y = _load_or_preprocess(args)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    trainer = TransformerTrainer(_build_config(args,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_epoch_fraction=args.sample_fraction,
        checkpoint_dir_transformer=args.checkpoint_dir,
        save_every_n_epoch=args.save_every,
    ))
    best_val_acc = trainer.train(AST(n_classes=len(np.unique(y))), X_train, y_train, X_val, y_val)
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

def cmd_predict(args) -> None:
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}"); sys.exit(1)
    try:
        predictor = AudioPredictor(model_path=args.model, device=args.device)
        predicted_class, top_probs, top_indices = predictor.predict_file(args.audio_file, top_k=args.top_k)
        labels = DatasetConfig().esc50_labels
        print("\n" + "=" * 60)
        print(f"Top {args.top_k} Predictions:")
        print("=" * 60)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            marker = "★" if idx == predicted_class else " "
            print(f"{marker} {i+1}. {labels[idx]:20s} - {prob*100:6.2f}%")
    except Exception as e:
        import traceback
        print(f"\nError during prediction: {e}")
        traceback.print_exc()
        sys.exit(1)

main()