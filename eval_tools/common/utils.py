from schema.eval_types import Family, HfSplit

def detect_family(model_name: str) -> Family:
    """Return the Family enum if any of its values is a substring of model_name."""
    model_name_lower = model_name.lower()
    for family in Family:
        if family.value and family.value.lower() in model_name_lower:
            return family
    return Family.NoneType_None

def detect_hf_split(split_str: str) -> HfSplit:
    """
    Determines the type of dataset split from a given string.
    
    Args:
        split_str (str): The input string to classify.
        
    Returns:
        HfSplit: One of "train", "test", or "validation".
    """
    s = split_str.strip().lower()
    
    if s == "test":
        return HfSplit.test
    elif "train" in s:
        return HfSplit.train
    else:
        return HfSplit.validation