import os
import seaborn as sn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils import set_seed
from trainer import *

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser()
    # Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
    parser.add_argument("--data_dir", type=str, default="./data/NEU-CLS-200/")
    parser.add_argument("--model_path", type=str, default="./models/best_resnet18_NEU-200.pth")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    config = parser.parse_args()
    return config


def load_model(num_classes, model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def eval_model(model, dataloader, idx_to_class):
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        y_labels = []
        y_predict = []
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_labels.extend(labels.data)
            y_predict.extend(preds)
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        y_labels = [idx_to_class[int(i.cpu().numpy())] for i in y_labels]
        y_predict = [idx_to_class[int(i.cpu().numpy())] for i in y_predict]
        classes = list(idx_to_class.values())
        conf_mat = confusion_matrix(y_labels, y_predict)
        print(conf_mat)
        print(classification_report(y_labels, y_predict))
        print("Acc on test set: {:4f}".format(epoch_acc))
        df_cm = pd.DataFrame(conf_mat, index=classes, columns=classes)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, cmap="YlGnBu", annot=True)
        plt.show()


def eval(data_dir, model_path, num_classes, batch_size=32):
    model = load_model(num_classes, model_path)
    model = model.to(device)

    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
    class_to_idx = image_datasets['val'].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['val']}

    eval_model(model, dataloaders_dict['val'], idx_to_class)


if __name__ == "__main__":
    config = get_config()
    batch_size = 32
    eval(config.data_dir, config.model_path, config.num_classes, config.batch_size)
