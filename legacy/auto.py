import os
# os.environ["WANDB_MODE"] = "offline"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import optuna
import wandb
import gc

from Image_Models import get_model_dict
from Image_Dataloader import Train_Loader, Valid_Loader, Test_Loader
from Optimizer_Scheduler import get_optimizer, get_scheduler
from Train_Process import train_model

# wandb.login(key="b35986593af0f00558f52c6b38699aea3e8e91c0")
NUM_CLASSES = 1
WORKERS = 8
TRIALS = 100
EPOCH = 50
FACTOR = 10

def inference(model, test_loader):
    results = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            probs = probs.detach().cpu().numpy()
            probs = (probs > 0.5).astype(np.int16)
            results.extend(probs)
    return results

def predict(model, stack_loader):
    results = []
    with torch.no_grad():
        for data in stack_loader:
            data = data[0].to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            probs = probs.detach().cpu().numpy()
            probs = (probs > 0.5).astype(np.int16)
            results.extend(probs)
    return results


if __name__ == "__main__":
    if not os.path.exists("./wandb_logs"):
        os.makedirs("./wandb_logs")
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print(f"GPU is available. Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Using CPU: {device}")
        
    # Load the CSV file
    base_dir = './'
    df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df['label'] = df['label'].map({"editada": 0, "real": 1}).astype(np.int16)
    # Path to the folder containing images
    image_folder = os.path.join(base_dir, 'Train')
    image_paths = [os.path.join(image_folder, img) for img in df['image']]
    labels = df['label'].values
    # Paths to test images
    df_test = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    test_image_folder = os.path.join(base_dir, 'Test')
    test_paths = [os.path.join(test_image_folder, img) for img in df_test['image']]

    # Split the data into training and validation sets
    train_paths, stacking_val_paths, train_labels, stacking_val_labels = train_test_split(image_paths, labels, test_size=0.4, random_state=42)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=2024)
    
    criterion = nn.BCEWithLogitsLoss()
    model_dict = get_model_dict(NUM_CLASSES)
    model_for_stacking = []
    
    train_loader = Train_Loader(train_paths, train_labels)
    # train_loader.set_transform_list(20, 0.2, 0.2, 0.2, 0.2)
    # train_loader.set_transform()
    # train_loader.set_dataset()
    stacking_val_loader = Valid_Loader(stacking_val_paths, stacking_val_labels)
    stacking_val_loader.set_transform_list()
    stacking_val_loader.set_transform()
    stacking_val_loader.set_dataset()
    val_loader = Valid_Loader(val_paths, val_labels)
    val_loader.set_transform_list()
    val_loader.set_transform()
    val_loader.set_dataset()
    test_loader = Test_Loader(test_paths)
    test_loader.set_transform_list()
    test_loader.set_transform()
    test_loader.set_dataset()
    
    batch_size = 96
    train_loader.set_batch_size(batch_size)
    stacking_val_loader.set_batch_size(batch_size)
    stacking_val_loader.set_dataloader(WORKERS)
    val_loader.set_batch_size(batch_size)
    val_loader.set_dataloader(WORKERS)
    test_loader.set_batch_size(batch_size)
    test_loader.set_dataloader(WORKERS)
    dataloaders = {'train': None, 'val': val_loader.valid_loader}
    print("Init Loader...")
    
    wandb.init(
        project="cidaut-ai-fake-scene-classification", 
        name="finalization",
        config={
            "epochs": EPOCH,
            "scheduler": "ReduceLROnPlateau"
        },
    )
    for model_name, model in model_dict.items():
        # print(f"{model_name}, batch_size = {batch_size}")
        def objective(trial):
            # Data augmentation parameters
            rotation_angle = trial.suggest_int('rotation_angle', 5, 30)
            brightness = trial.suggest_float('brightness', 0.1, 0.3)
            contrast = trial.suggest_float('contrast', 0.1, 0.3)
            saturation = trial.suggest_float('saturation', 0.1, 0.3)
            hue = trial.suggest_float('hue', 0.1, 0.3)
            
            train_loader.set_transform_list(
                rotation_angle, brightness, 
                contrast, saturation, hue
            )
            train_loader.set_transform()
            train_loader.set_dataset()
            train_loader.set_dataloader(WORKERS)
            dataloaders["train"] = train_loader.train_loader
            
            # Define hyperparameters to tune
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            factor = trial.suggest_float('factor', 0.1, 0.5)
            patience = trial.suggest_int('patience', 3, 10)
            
            optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
            scheduler = get_scheduler(optimizer, factor, patience)
            # Train the model
            try:
                best_auc, _, _ = train_model(
                    model, criterion, 
                    optimizer, scheduler,
                    dataloaders, device, 
                    num_epochs=EPOCH, 
                    trial=trial
                )
            except Exception as e:
                print(f"Error during training in trial {trial.number}: {e}")
                raise optuna.TrialPruned()  # You can prune the trial if desired
            return best_auc
        
        storage_name = f"sqlite:///{model_name}_study.db"
        study = optuna.create_study(direction="maximize", storage=storage_name, study_name=f"{model_name}_study", load_if_exists=True)
        study.optimize(objective, n_trials=TRIALS, n_jobs=1) 
        print(f"Best trial of {model_name}:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        train_loader.set_transform_list(
            trial.params['rotation_angle'], 
            trial.params['brightness'], 
            trial.params['contrast'], 
            trial.params['saturation'], 
            trial.params['hue']
        )
        train_loader.set_transform()
        train_loader.set_dataset()
        train_loader.set_dataloader(WORKERS)
        dataloaders["train"] = train_loader.train_loader
        best_optimizer = get_optimizer(
            model, trial.params['optimizer'], 
            trial.params['lr'], trial.params['weight_decay']
        )
        best_scheduler = get_scheduler(
            best_optimizer, 
            trial.params['factor'], trial.params['patience']
        )
        best_auc, train_scores, validation_scores = train_model(
            model, criterion, 
            best_optimizer, best_scheduler,
            dataloaders, device,
            num_epochs=int(EPOCH * FACTOR)
        )
        val_score = validation_scores[-1]
        print(f"Final train scores: {train_scores[-1]} val scores: {val_score}")
        if val_score > 0.70:
            model.load_state_dict(torch.load('best_model.pth', weights_only=True))
            model_for_stacking.append(model.eval())
        # # optuna.delete_study(study_name=f"{model_name}_study", storage=storage_name)
        del study
        gc.collect()
        
    wandb.finish()
    
    print("Stacking...")
    meta_data = [predict(model, stacking_val_loader.valid_loader) for model in model_for_stacking]
    stacked_preds = np.column_stack(meta_data)
    meta_model = LogisticRegression()
    meta_model.fit(stacked_preds, stacking_val_labels)
    
    meta_test = [inference(model, test_loader.test_loader) for model in model_for_stacking]
    meta_test = np.column_stack(meta_test)
    final_preds = meta_model.predict(meta_test)
    df_test['label'] = final_preds
    df_test.to_csv('submission.csv', index=False)
    print(df_test.head())
    print(df_test.tail())
    print("Done!")
        