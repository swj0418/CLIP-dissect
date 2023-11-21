import os
import pickle

import torch
import pandas as pd
import torchvision.datasets

import dnnlib
import legacy
from torchvision import datasets, transforms, models

DATASET_ROOTS = {"imagenet_val": "data/ImageNet_val/",
                "broden": "data/broden1_224/images/"}


class GANDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(1004)
        self.codes = torch.load(os.path.join(self.root, 'class_index.pt'), map_location='cpu')

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        target = self.codes[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif target_name == 'resnet50_imagenet_random_split0':
        target_model = models.resnet50(num_classes=500).to(device)
        state_dict = torch.load(f'data/{target_name}.ckpt', map_location=device)['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('model.'):
                new_state_dict[key[6:]] = state_dict[key]

        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    elif target_name == 'resnet50_imagenet_random_split1':
        target_model = models.resnet50(num_classes=500).to(device)
        state_dict = torch.load(f'data/{target_name}.ckpt', map_location=device)['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('model.'):
                new_state_dict[key[6:]] = state_dict[key]

        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    elif target_name == 'resnet50_imagenet_artificial':
        target_model = models.resnet50(num_classes=550).to(device)
        state_dict = torch.load(f'data/{target_name}.ckpt', map_location=device)['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('model.'):
                new_state_dict[key[6:]] = state_dict[key]

        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    elif target_name == 'resnet50_imagenet_natural':
        target_model = models.resnet50(num_classes=450).to(device)
        state_dict = torch.load(f'data/{target_name}.ckpt', map_location=device)['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('model.'):
                new_state_dict[key[6:]] = state_dict[key]

        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = models.ResNet50_Weights.DEFAULT.transforms()
    elif target_name == 'imagenet256':
        with dnnlib.util.open_url(f'data/{target_name}.pkl') as f:
            target_model = pickle.load(f)['G_ema']
            target_model = target_model.eval().requires_grad_(False).to(device)

        print(target_model)
        preprocess = None
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    
    target_model.eval()
    return target_model, preprocess


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    elif dataset_name == 'gan_probe':
        data = GANDataset(root=f'data/gan_val', transform=preprocess)
        
    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass