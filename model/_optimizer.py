from torch import optim



def get_optimizer(params, **kwargs):
    """_summary_

    Args:
        params (_type_): model.parameters()

    Raises:
        NameError: _description_

    Returns:
        _type_: _description_
    """
    name = kwargs["name"] if "setting" in kwargs else "adam"
    setting = kwargs["setting"] if "setting" in kwargs else {}
    
    
    if name == "adam":
        return optim.Adam(
            params,
            **setting
        )
        
    if name == "adamW":
        return optim.AdamW(
            params,
            **setting
        )
    
    if name == "SGD":
        return optim.SGD(
            params,
            **setting
        )
        
    raise NameError("No such optimizer exists")