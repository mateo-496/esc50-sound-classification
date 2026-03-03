import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Sequence

from src.config.config import TrainConfig
from src.data.dataset import FullTFPatchesDataset, RandomPatchDataset

config = TrainConfig()

class TransformerTrainer:
    def __init__(self, config: TrainConfig = config) -> None:
        self.config = config

    def train(
        self,
        model: nn.Module,
        X_train: Sequence[np.ndarray],
        y_train: Sequence[int],
        X_val: Sequence[np.ndarray],
        y_val: Sequence[int],
    ) -> float:
        device = self.config.device
        os.makedirs(self.config.checkpoint_dir_transformer, exist_ok=True)
        model.to(device)

        if self.config.use_all_patches:
            train_dataset = FullTFPatchesDataset(X_train, y_train)
            print(f"\n{'='*60}\nUsing ALL PATCHES method\n{'='*60}")
        else:
            train_dataset = RandomPatchDataset(X_train, y_train)
            print(f"\n{'='*60}\nUsing RANDOM PATCHES method\n{'='*60}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            FullTFPatchesDataset(X_val, y_val),
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=True
        )

        total_patches = len(train_dataset)
        patches_per_epoch = int(total_patches * self.config.samples_per_epoch_fraction)
        batches_per_epoch = patches_per_epoch // self.config.batch_size
        print(f"Total available patches: {total_patches:,}")
        print(f"Patches per epoch ({self.config.samples_per_epoch_fraction}): {patches_per_epoch:,}")
        print(f"Batches per epoch: {batches_per_epoch:,}\n{'='*60}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=1e-6
        )

        start_epoch = 0
        best_val_acc = 0.0
        best_epoch = 0
        training_history: dict = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'epochs': []}

        if self.config.resume_from and os.path.exists(self.config.resume_from):
            print(f"Resuming from checkpoint: {self.config.resume_from}")
            checkpoint = torch.load(self.config.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            training_history = checkpoint['history']
            print(f"Resuming from epoch {checkpoint['epoch']}, best val acc: {best_val_acc:.4f}\n")

        start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            if epoch < self.config.warmup_epochs:
                lr = self.config.lr * (epoch + 1) / self.config.warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            model.train()
            train_loss, correct, total, batches_processed = 0.0, 0, 0, 0

            for xb, yb in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
                if batches_processed >= batches_per_epoch:
                    break
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * xb.size(0)
                correct += out.argmax(dim=1).eq(yb).sum().item()
                total += yb.size(0)
                batches_processed += 1

            train_loss /= total
            train_acc = correct / total

            if epoch >= self.config.warmup_epochs:
                scheduler.step()

            val_acc = self._validate(model, X_val, y_val, device, epoch)

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch

            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['epochs'].append(epoch + 1)

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
                f"Val acc: {val_acc:.4f} (best: {best_val_acc:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if (epoch + 1) % self.config.save_every_n_epoch == 0 or is_best:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': training_history,
                }
                torch.save(checkpoint, os.path.join(self.config.checkpoint_dir_transformer, 'latest_checkpoint.pt'))
                if is_best:
                    torch.save(checkpoint, os.path.join(self.config.checkpoint_dir_transformer, 'best_model.pt'))
                    print(f'✓ Saved best model with val_acc: {val_acc:.4f}')

                with open(os.path.join(self.config.checkpoint_dir_transformer, 'training_history.json'), 'w') as f:
                    json.dump(training_history, f, indent=2)

            if epoch - best_epoch > 30:
                print(f'\nEarly stopping: no improvement for 30 epochs')
                break

        elapsed = time.time() - start_time
        print(f'\n{"="*60}')
        print(f'Training completed in {elapsed/3600:.2f} hours')
        print(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')
        print(f'{"="*60}')

        final_dir = "models/saved"
        os.makedirs(final_dir, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc}, 
                   os.path.join(final_dir, "final_transformer.pt"))

        return best_val_acc

    def _validate(self, model: nn.Module, X_val: Sequence[np.ndarray], y_val: Sequence[int], device: str, epoch: int) -> float:
        from src.config.config import DatasetConfig
        cfg = DatasetConfig()
        model.eval()
        correct = 0
        for i in tqdm.tqdm(range(len(y_val)), desc=f"Epoch {epoch+1} Val", leave=False):
            spec = X_val[i]
            n_frames = spec.shape[0]
            if n_frames < cfg.cnn_input_length:
                spec = np.pad(spec, ((0, cfg.cnn_input_length - n_frames), (0, 0)), mode='constant')
                n_frames = cfg.cnn_input_length
            patches = np.stack([
                spec[s:s + cfg.cnn_input_length]
                for s in range(0, n_frames - cfg.cnn_input_length + 1)
            ])[:, np.newaxis]
            with torch.no_grad():
                out = model(torch.tensor(patches, dtype=torch.float32).to(device)).mean(dim=0)
            if out.argmax().item() == y_val[i]:
                correct += 1
        return correct / len(y_val)