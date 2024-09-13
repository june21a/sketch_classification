from torch import optim



def get_optimizer(name, params, **kwargs):
    """_summary_

    Args:
        name (_type_): optimizer name
        params (_type_): model.parameters()

    Raises:
        NameError: _description_

    Returns:
        _type_: _description_
    """
    if name == "adam":
        return optim.Adam(
            params,
            **kwargs
        )
        
    if name == "adamW":
        return optim.AdamW(
            params,
            **kwargs
        )
    
    if name == "SGD":
        return optim.SGD(
            params,
            **kwargs
        )
        
    raise NameError("No such optimizer exists")