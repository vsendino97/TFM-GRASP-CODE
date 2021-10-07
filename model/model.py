import torch.nn as nn
import torchvision
import torch
from base import BaseModel

class ImageModel(BaseModel):
    def __init__(self, model_name, num_categories, tune_layers, class_layer):
        super(ImageModel, self).__init__()

        #model_name = "im_resnet50"
        if (model_name == "im_resnet18"):
            self.model = torchvision.models.resnet18(pretrained=True)
        else:
            print("Resnet50")
            self.model = torchvision.models.resnet50(pretrained=True)

        for name, param in self.model.named_parameters():
            print (name, param.data)

        self.transform_params = {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 8,
            "sampling_rate": 8,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

        freeze_trainable_layers(self.model, tune_layers)
        self.classifier_layer = setup_class_layer("images", num_categories)

    def forward(self, inputs):
        x = self.model(inputs)
        # Pooling and final linear layer
        x = x.flatten(start_dim=1)
        x = self.classifier_layer(x)
        return x


class VideoModel(BaseModel):
    def __init__(self, model_name, num_categories, tune_layers, class_layer):
        super(VideoModel, self).__init__()

        if (model_name == "r2plus1d"):
            self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
            self.transform_params = {
                "side_size": 112,
                "crop_size": 112,
                "num_frames": 8,
                "sampling_rate": 8,
                "mean": [0.45, 0.45, 0.45],
                "std": [0.225, 0.225, 0.225]
            }
        elif (model_name == "slow_r50"):
            self.model = torch.hub.load('facebookresearch/pytorchvideo', model="slow_r50", pretrained=True) 
            self.transform_params = {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 8,
                "sampling_rate": 8,
                "mean": [0.45, 0.45, 0.45],
                "std": [0.225, 0.225, 0.225]
            }
        elif (model_name == "slowfast_r50"):
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            self.transform_params = {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 8,
                "sampling_rate": 8,
                "mean": [0.45, 0.45, 0.45],
                "std": [0.225, 0.225, 0.225]

            }

        elif (model_name == "x3d_m"):
            self.model = torch.hub.load('facebookresearch/pytorchvideo', model="x3d_m", pretrained=True)
            self.transform_params = {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
                "mean": [0.45, 0.45, 0.45],
                "std": [0.225, 0.225, 0.225]
            }

        else:
            print("Model not implemented")
            exit()
        freeze_trainable_layers(self.model, tune_layers)
        self.classifier_layer = setup_class_layer(class_layer, num_categories)

    def forward(self, inputs):
        x = self.model(inputs)
        x = self.classifier_layer(x)
        return x


def reset_weights(m):
    """
        Try resetting model weights to avoid
        weight leakage.
    """
    for layer in m.children():
        print(layer)
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def freeze_trainable_layers(m, tr_layers):
    """
        Freeze {fr_layers} convolutional layers from model m,
        so they are not trainable.
    """

    num_layers = 0
    for name, param in m.named_parameters():
        num_layers += 1
    
    if (tr_layers == 0):
        for param in m.parameters():
            param.requires_grad = False
        print("Frozen all layers")
    elif (tr_layers > 0 and tr_layers <= num_layers):
        fr_layers = num_layers - tr_layers
        layer_ct = 0
        for name, param in m.named_parameters():
            if layer_ct < fr_layers:
                param.requires_grad = False
            else: 
                param.requires_grad = True
            layer_ct += 1
        print(f"Frozen the first {fr_layers} layers from {layer_ct} in total.")

        #for name, param in m.named_parameters():
        #    print(name, param.requires_grad)
    else:
        print("Invalid number of convolutional layers to train")
        exit()
    
def setup_class_layer(class_layer, num_categories):
    """
        Setup the classification layer architecture.
    """
    if (class_layer == "small"):
        return nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_categories)
        )

    elif (class_layer == "full"):
        return nn.Sequential(
            nn.Linear(400, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_categories)
        )

    elif (class_layer == "images"):
        return nn.Sequential(
            #nn.Linear(1000, 256),
            #nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            #nn.Linear(256, 128),
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 64),
            #nn.ReLU(inplace=True),
            #nn.Linear(64, num_categories)
            nn.Linear(1000,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_categories)
        )
    else:
        print("Invalid classification layer.")
        exit()


