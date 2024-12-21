from torch import optim


def get_schedular(optimizer, **kwargs):
    name = kwargs["name"] if "setting" in kwargs else "cosine"
    setting = kwargs["setting"] if "setting" in kwargs else {}
    
    if name == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            **setting
        )
    elif name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **setting
        )
    else:
        raise NameError(f"{name} is not defined")
    
    return scheduler