
import torch
import torch.nn as nn
from torchvision import models

from .alexnet_micro import alexnet as alexnet_m
from .resnet_micro import resnet18 as resnet18_m
from .resnet_micro import resnet34 as resnet34_m
from .densenet_micro import densenet121 as densenet121_m
from .value_model import Value_model
from .mlp import MLP, Simple
from .cnnn import FlexibleCNN
from .vit import VisionTransformer
from transformers import ViTForImageClassification

def load_model(name, outputsize, pretrained=None, args=None, nb_class=10):

    if pretrained:
        pretrained = True
    else:
        pretrained = False
    # 不用去模型的定义处修改outputsize，直接在这里就行
    if name.lower() in 'alexnet_micro':
        model = alexnet_m(pretrained=pretrained)
        model.classifier = nn.Linear(256, outputsize)
    elif name.lower() in 'resnet18_micro':
        model = resnet18_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'resnet34_micro':
        model = resnet34_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'densenet121_micro':
        model = densenet121_m(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, outputsize)
    
    elif name.lower() in 'mlp':
        if args.dataset_name == 'mnist':
            model = MLP(input_channels=1, image_size=args.image_size, num_classes=nb_class)
        else:
            model = MLP(input_channels=3, image_size=args.image_size, num_classes=nb_class)
    elif name.lower() in 'simple':
        if args.dataset_name == 'mnist':
            model = Simple(input_channels=1, image_size=args.image_size, num_classes=nb_class)
        else:
            model = Simple(input_channels=3, image_size=args.image_size, num_classes=nb_class)
    elif name.lower() in 'cnn':
        model = FlexibleCNN(num_classes=nb_class)
    elif name.lower() in 'vit':
        if pretrained:
            model = ViTForImageClassification.from_pretrained('/mnt/csp/mmvision/home/lwh/DLS/models--google--vit-base-patch16-224/snapshots/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3', num_labels=nb_class, ignore_mismatched_sizes=True)
        else:
            model = VisionTransformer(args.image_size, 16, nb_class, 512, 6, 8, 1024, dropout=0, emb_dropout=0)
    return model

def load_valuemodel(input_size, hidden_size, num_classes):
    model = Value_model(input_size, hidden_size, num_classes)
    return model


