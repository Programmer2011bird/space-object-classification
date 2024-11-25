from torchvision import models
from torch.nn import Linear
from torch import argmax
import dataloader


NUM_CLASSES: int = 4
LEARNING_RATE: float = 0.001
EPOCHS: int = 3

MODEL = models.resnet50()
MODEL.fc = Linear(MODEL.fc.in_features, NUM_CLASSES)

train_loader, test_loader, class2idx = dataloader.get_dataloader()
# TODO: write the training code after finishing traintest.py
for img, label in train_loader:
    print(img, label)
    output = MODEL(img)
    print(output)
    print(argmax(output, dim=1))
    
    break
