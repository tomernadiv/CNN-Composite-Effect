import torch
import torch.nn as nn

class VGG16FeatureNet(nn.Module):
    def __init__(self, original_model, layer_index):
        super(VGG16FeatureNet, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:layer_index+1])
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x