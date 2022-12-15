import torch_geometric.transforms as T


def get_transforms_list(type: str = "cartesian"):
    if type is "cartesian":
        return [
            T.GCNNorm(),
            T.Cartesian(cat=False),
            T.NormalizeScale(),
            T.NormalizeFeatures(),
        ]
    elif type is "distance":
        return [
            T.GCNNorm(),
            T.Distance(cat=False),
            T.NormalizeScale(),
            T.NormalizeFeatures(),
        ]
    elif type is "localcartesian":
        return [
            T.GCNNorm(),
            T.LocalCartesian(cat=False),
            T.NormalizeScale(),
            T.NormalizeFeatures(),
        ]
    elif type is "slic":
        return [
            T.ToSLIC(),
            T.GCNNorm(),
            T.Cartesian(cat=False),
            T.NormalizeScale(),
            T.NormalizeFeatures(),
        ]


def get_transforms(type: str = "cartesian"):
    """
    Returns a composed pipeline of transformations

    params:
    type - {string} - Options default, distance, localcartesian, slic

    Passing in the type string returns the list of transforms associated with
    that string.
    Default - GCNNorm, Cartesian, Normalize Scale and Features
    Distance - GCNNorm, Distance, Normalize Scale and Features
    Local Cartesian - GCNNorm, Local Cartesian, Normalize Scale and Features
    SLIC - SLIC, GCNNorm, Cartesian, Normalize Scale and Features

    NOTE: SLIC can only be used with torchvision datasets, converts them to a graph
    """
    transforms_list = get_transforms_list(type)
    return T.Compose(transforms_list)
