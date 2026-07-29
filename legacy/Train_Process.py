from tqdm import trange 
import torch
from sklearn.metrics import roc_auc_score
import wandb
import optuna

def train_model(
    model, criterion, 
    optimizer, scheduler,
    dataloaders, device,
    num_epochs,
    trial=None
):
    train_scores, validation_scores = [], []
    best_model_path = 'best_model.pth'
    best_auc = -1
    model = model.to(device)
    for epoch in trange(1, num_epochs + 1):
        """
        Train
        """
        model.train()
        all_preds = []
        all_labels = []
        for inputs, labels in dataloaders['train']:
            labels = labels.unsqueeze(1).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient of {name}")

        # Statistics
        probs = torch.sigmoid(outputs)  # Assuming binary classification
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        train_scores.append(roc_auc_score(all_labels, all_preds))
        
        """
        Valid
        """
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                labels = labels.unsqueeze(1).float()
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(dataloaders['val'])
        scheduler.step(val_loss)
        
        # Statistics
        probs = torch.sigmoid(outputs)  # Assuming binary classification
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        auc = roc_auc_score(all_labels, all_preds)
        validation_scores.append(auc)
        
        wandb.log({
            "train_auc": train_scores[-1],
            "val_auc": validation_scores[-1],
        })
        if trial:
            trial.report(validation_scores[-1], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if auc > best_auc:
            best_auc = auc
            if trial is None:
                torch.save(model.state_dict(), best_model_path)
    return best_auc, train_scores, validation_scores