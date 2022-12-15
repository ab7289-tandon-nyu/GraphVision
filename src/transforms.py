import torch_geometric.transforms as TG
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment


def get_transforms_list(type: str = "cartesian"):
    if type == "cartesian":
        return [
            TG.GCNNorm(),
            TG.Cartesian(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
    elif type == "distance":
        return [
            TG.GCNNorm(),
            TG.Distance(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
    elif type == "localcartesian":
        return [
            TG.GCNNorm(),
            TG.LocalCartesian(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
    elif type == "mnist-slic":
        train_transform = [
            transforms.ToTensor(),
            TG.ToSLIC(),
            TG.GCNNorm(),
            TG.Cartesian(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
        return train_transform
    elif type == "cifar10-slic":
        train_transform = [
            transforms.AutoAugment(policy=autoaugment.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            TG.ToSLIC(),
            TG.GCNNorm(),
            TG.Cartesian(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
        test_transform = [
            transforms.ToTensor(),
            TG.ToSLIC(),
            TG.GCNNorm(),
            TG.Cartesian(cat=False),
            TG.NormalizeScale(),
            TG.NormalizeFeatures(),
        ]
        return train_transform, test_transform


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
    mnist-slic - ToTensor, SLIC, GCNNorm, Cartesian, Normalize Scale and Features
    cifar10-slic - AutoAugment, ToTensor, ToSLIC, Cartesion, Normalize and Scale features

    NOTE: SLIC can only be used with torchvision datasets, converts them to a graph
    """
    transforms_list = get_transforms_list(type)
    if "slic" in type:
        return TG.Compose(transforms_list[0]), TG.Compose(transforms_list[1])
    else:
        return TG.Compose(transforms_list)
