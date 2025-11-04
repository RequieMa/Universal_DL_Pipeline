import torch.optim as optim

def get_optimizer(model, optimizer_name, lr, weight_decay):
    return getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, factor, patience):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
