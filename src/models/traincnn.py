import os
import torch
import tqdm
import json
import numpy as np
from torch.utils.data import DataLoader
from typing import Sequence

from src.models.predict import AudioPredictor
from src.data.dataset import FullTFPatchesDataset, RandomPatchDataset
from src.config.config import TrainConfig

config = TrainConfig()

class CNNTrainer:
    def __init__(self, config: TrainConfig = config) -> None:
        self.config = config

    def train_cnn(
        self,
        model: torch.nn.Module,
        X_train: Sequence[np.ndarray], 
        y_train: Sequence[int],
        X_val: Sequence[np.ndarray], 
        y_val: Sequence[int],
        fold_num: int,
    ) -> float:
        device = self.config.device
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        model.to(device)
        
        if self.config.use_all_patches:
            train_dataset = FullTFPatchesDataset(X_train, y_train)
            print(f"\n{'='*60}\nUsing ALL PATCHES method\n{'='*60}")
        else:
            train_dataset = RandomPatchDataset(X_train, y_train)
            print(f"\n{'='*60}\nUsing ALL PATCHES method\n{'='*60}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        total_patches = len(train_dataset)
        patches_per_epoch = int(total_patches * self.config.samples_per_epoch_fraction)
        batches_per_epoch = patches_per_epoch // self.config.batch_size

        print(f"Total available patches: {total_patches:,}")
        print(f"Patches per epoch ({self.config.samples_per_epoch_fraction}): {patches_per_epoch:,}")
        print(f"Batches per epoch: {batches_per_epoch:,}\n{'='*60}\n")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.features.parameters(), 'weight_decay': 0.0},
            {'params': model.classifier.parameters(), 'weight_decay': 0.001}
        ], lr=self.config.lr)
        
        start_epoch = 0
        best_val_acc = 0.0
        training_history: dict = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'epochs': []}

        if self.config.resume_from and os.path.exists(self.config.resume_from):
            print(f"Resuming from checkpoint: {self.config.resume_from}")
            checkpoint = torch.load(self.config.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            training_history = checkpoint['history']
            print(f"Resuming from epoch {checkpoint['epoch']}, best val acc: {best_val_acc:.4f}\n")

        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            train_loss, correct, total, batches_processed = 0.0, 0, 0, 0
            
            for xb, yb in tqdm.tqdm(train_loader, f"Epoch {epoch+1} Train", leave=False):
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
                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                batches_processed += 1
            
            train_loss /= total
            train_acc = correct / total
            
            model.eval()
            val_correct = 0
            val_total = len(y_val)
            
            for i in tqdm.tqdm(range(val_total), desc=f"Epoch {epoch+1} Val", leave=False):
                spec = X_val[i]
                true_label = y_val[i]
                
                pred_label = self._predict_val(model, spec, device)                
                
                if pred_label == true_label:
                    val_correct += 1
            
            val_acc = val_correct / val_total

            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['epochs'].append(epoch + 1)

            is_best = val_acc > best_val_acc

            if is_best:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pt")
            
            print(
                f"Fold {fold_num} | Epoch {epoch+1}/{self.config.epochs} | "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
                f"Val acc: {val_acc:.4f} (best: {best_val_acc:.4f})"
            )

            if (epoch + 1) % self.config.save_every_n_epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'history': training_history,
                    'config': {
                        'batch_size': self.config.batch_size,
                        'lr': self.config.lr,
                        'total_patches': total_patches,
                        'patches_per_epoch': patches_per_epoch,
                    }
                }
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                
                if is_best:
                    best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                    torch.save(checkpoint, best_path)

                latest_path = os.path.join(self.config.checkpoint_dir, "latest_checkpoint.pt")
                torch.save(checkpoint, latest_path)
                
                history_path = os.path.join(self.config.checkpoint_dir, "training_history.json")
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=2)

        final_model_dir = "models/saved"
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = os.path.join(final_model_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'config': {
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'epochs': self.config.epochs,
            }
        }, final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}")

        return best_val_acc

    def train_k_fold_cnn(
        self,
        model_class: type,
        X: Sequence[np.ndarray],
        y: Sequence[int],
    ) -> tuple[list[float], float]:

        X_arr = np.array(X)
        y_arr = np.array(y)
        n_samples = len(y_arr)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_sizes = (n_samples // 5) * np.ones(5, dtype=int)
        fold_sizes[:n_samples % 5] += 1
        current = 0
        fold_accuracies: list[float] = []

        for fold_num, fold_size in enumerate(fold_sizes, 1):
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            current = stop

            X_train, y_train = X_arr[train_idx].tolist(), y_arr[train_idx]
            X_val, y_val = X_arr[val_idx].tolist(), y_arr[val_idx]

            print(f"\n{'='*80}\nFOLD {fold_num}/5 | Train: {len(X_train)}, Val: {len(X_val)}\n{'='*80}\n")

            model = model_class()
            best_acc = self.train_cnn(
                model=model,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                fold_num=fold_num,
            )
            fold_accuracies.append(best_acc)
            print(f"\nFold {fold_num} Best Accuracy: {best_acc:.4f}\n")

        mean_acc = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies))

        print(f"\n{'='*80}\nFINAL 5-FOLD CV RESULTS\nFold Accuracies: {fold_accuracies}\nMean: {mean_acc:.4f} ± {std_acc:.4f}\n{'='*80}\n")

        results_path = os.path.join(self.config.checkpoint_dir, "5fold_cv_results.json")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({'fold_accuracies': fold_accuracies, 'mean_accuracy': mean_acc, 'std_accuracy': std_acc}, f, indent=2)

        return fold_accuracies, mean_acc

    def _predict_val(self, model: torch.nn.Module, spec: np.ndarray, device: str) -> int:
        from src.config.config import DatasetConfig
        cfg = DatasetConfig()
        n_frames = spec.shape[0]
        if n_frames < cfg.cnn_input_length:
            spec = np.pad(spec, ((0, cfg.cnn_input_length - n_frames), (0, 0)), mode="constant")
            n_frames = cfg.cnn_input_length
        patches = np.stack([
            spec[s:s + cfg.cnn_input_length]
            for s in range(0, n_frames - cfg.cnn_input_length + 1)
        ])[:, np.newaxis]
        patches_t = torch.tensor(patches, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(patches_t).mean(dim=0)
        return out.argmax().item()