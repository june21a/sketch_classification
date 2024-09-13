from torch import nn


def get_loss(name= "cross_entropy"):
    """_summary_

    Args:
        name (str, optional): name of loss function. Defaults to "cross_entropy".

    Returns:
        torch.nn.modules.loss: 해당 이름의 loss function
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss()




############################################################
## focal loss 등 필요한 loss 추가 구현 후 get_loss에 적용하기