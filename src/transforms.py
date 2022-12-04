import torch_geometric.transforms as T


def get_transforms_list(type: str = "cartesian"):
    if type is "default":
        return [
            T.NormalizeScale(),
            T.Cartesian(cat=True),
        ]
    elif type is "distance":
        return [
            T.NormalizeScale(),
            T.Distance(cat=False),
        ]
    elif type is "localcartesian":
        return [
            T.NormalizeScale(),
            T.LocalCartesian(cat=False),
        ]


def get_transforms(type: str = "cartesian"):
    """
    Returns a composed pipeline of transformations
    """
    transforms_list = get_transforms_list(type)
    return T.Compose(transforms_list)
