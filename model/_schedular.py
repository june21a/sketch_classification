from torch import optim


def get_schedular(name, optimizer, steps_per_epoch, **kwargs):
    if name == "steplr":
        scheduler_step_size = 30  # 매 30step마다 학습률 감소
        scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소
        

        # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
        epochs_per_lr_decay = 2
        scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
    elif name == "cosine":
        T_max = 10
        eta_min = 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    return scheduler