import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import set_seed
from trainer import *

set_seed(42)

# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
"""Data 1"""
# data_dir = "./data/NEU-CLS-64/"
# pretrained_path = None
# num_classes = 9  # for NEU-CLS-64
# model_name = "resnet"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

"""Data 2"""
data_dir = "./data/NEU-CLS-200/"
pretrained_path = "models/best_resnet18_ImageNet_NEU-64.pth"
# pretrained_path = "models/best_resnet18_ImageNet.pth"
# pretrained_path = "models/best_resnet18_NEU-64.pth"
num_classes = 6  # for NEU-CLS-200
model_name = "resnet_NEU-64"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

batch_size = 8
lr = 0.0001
num_epochs = 25

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

"""
Fine Tune with ImageNet 
"""
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True,
                                        pretrained_path=pretrained_path)
print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.AdamW(params_to_update, lr=lr)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                             is_inception=(model_name == "inception"))

# torch.save(model_ft.state_dict(), "models/resnet18_ImageNet_NEU-64.pth")  # save
torch.save(model_ft.state_dict(), "models/resnet18_ImageNet_NEU-64_NEU-200.pth")  # save
# torch.save(model_ft.state_dict(), "models/resnet18_ImageNet_NEU-64_NEU-200_fixed.pth")  # save
# torch.save(model_ft.state_dict(), "models/resnet18_NEU-64_NEU-200.pth")  # save
# torch.save(model_ft.state_dict(), "models/resnet18_NEU-64_NEU-200_fixed.pth")  # save
# torch.save(model_ft.state_dict(), "models/resnet18_ImageNet_NEU-200.pth")  # save
# torch.save(model_ft.state_dict(), "models/resnet18_ImageNet_NEU-200_fixed.pth")  # save

"""
Comparison with Model Trained from Scratch
"""
print("=" * 100)
# Initialize the non-pretrained version of the model used for this run
scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
params = [p for p in scratch_model.parameters() if p.requires_grad]
scratch_optimizer = optim.AdamW(params, lr=lr)
# scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scratch_criterion = nn.CrossEntropyLoss()
scratch_model, scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
                                          num_epochs=num_epochs, is_inception=(model_name == "inception"))

torch.save(scratch_model.state_dict(), "models/resnet18_NEU-200.pth")  # save
# torch.save(scratch_model.state_dict(), "models/resnet18_NEU-64.pth")  # save

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [float(h.cpu().numpy()) for h in hist]
print(ohist)
shist = [float(h.cpu().numpy()) for h in scratch_hist]
print(shist)

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.legend()
