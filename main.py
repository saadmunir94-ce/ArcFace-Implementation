# Import libraries
import numpy as np
import os
from model import *
from torch.utils.data import DataLoader
from data import Face_Dataset
import matplotlib.pyplot as plt
from trainer import train_model
from torchvision import models

# Load the pretrained model
pretrained_model = models.resnet50(pretrained=True)
print(pretrained_model)
# Freeze weights except for the last layer
modules = list(pretrained_model.children())[:-1]
# Formulate the deep feature extractor
ft_extractor = nn.Sequential(*modules)
for p in ft_extractor.parameters():
    p.requires_grad = False
# Set the number of classes
num_classes = 18
# Read Data
# Set the directories
print(os.getcwd())
train_dir = "./train_data"
test_dir = "./test_data"
# Read images
train_images = np.load(os.path.join(train_dir, "train_images.npy"))
test_images = np.load(os.path.join(test_dir, "test_images.npy"))
# Read labels
train_labels = np.load(os.path.join(train_dir, "train_labels.npy"))
test_labels = np.load(os.path.join(test_dir, "test_labels.npy"))
# Create the Face Datasets
train_set = Face_Dataset(data={"images": train_images, "labels": train_labels}, mode="train")
test_set = Face_Dataset(data={"images": test_images, "labels": test_labels}, mode="val")
train_sampler = train_set.create_sampler()
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False)
valid_loader = DataLoader(test_set, batch_size=batch_size)
num_epochs = 25
is_softmax = False
model = Model(ft_extractor, num_classes, is_softmax).double()

metric_fc = ArcFace(emb_size=512, num_classes=num_classes).double()
is_print = True
if is_print:
    for p in model.parameters():
        print(p.requires_grad)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if not is_softmax:
    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                                 lr=1e-3)
    print("Optimizer optimizing both ArcFace and model parameters")
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Optimizer only doing model parameters")
criterion = nn.CrossEntropyLoss()
img, label = train_set[0]
print(img.shape)
print(label.shape)
for img, label in train_loader:
    print(img.shape)
    print(label.shape)
train_losses, val_losses, save_dir, metrics = train_model(model=model, criterion=criterion, optimizer=optimizer, dataloaders={"train": train_loader, "val": valid_loader}, metric_fc=metric_fc, is_softmax=is_softmax, num_epochs=num_epochs)
# Plot Loss Curve
fig = plt.figure()
plt.plot(np.arange(1, len(train_losses)+1), train_losses, label="train loss")
plt.plot(np.arange(1, len(val_losses)+1), val_losses, label="valid loss")
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(fname=os.path.join(save_dir, "loss_curve.jpg"))
plt.show()
# Plot Validation Accuracy
fig = plt.figure()
plt.plot(np.arange(1, len(train_losses)+1), metrics["acc"], label="Valid Accuracy")
plt.xlabel("No of Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig(fname=os.path.join(save_dir, "valid_acc.jpg"))
plt.show()
#Plot Validation F1 score
fig = plt.figure()
plt.plot(np.arange(1, len(train_losses)+1), metrics["f1"], label="Valid F1")
plt.xlabel("No of Epochs")
plt.ylabel("F1 score")
plt.legend()
plt.savefig(fname=os.path.join(save_dir, "valid_f1.jpg"))
plt.show()