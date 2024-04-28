import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from matplotlib import pyplot as plt
from datetime import datetime

from resnet import ResidualNetwork
from model_trainer import ModelTrainer
from classification_dataset import ClassificationDataset

# ==============================================================================
DATASET_NAME = "EuroSAT_RGB"
DATASET_URL  = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
CLASSES_LIST = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                'River', 'SeaLake']

MODELS_PATH  = "./models"
DATASET_PATH = "./dataset"

NUM_OF_EPOCHS = 80
BATCH_SIZE    = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY  = 10e-5
RES_BLOCKS_PER_LAYER = [2, 2, 2, 2]
# ==============================================================================

def train_model():
    os.makedirs(MODELS_PATH, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResidualNetwork(RES_BLOCKS_PER_LAYER, len(CLASSES_LIST))
    
    dataset = ClassificationDataset(dataset_root_dir=DATASET_PATH, dataset_name=DATASET_NAME, 
                                    dataset_url=DATASET_URL, classes_labels=CLASSES_LIST)

    train_dataset, valid_dataset, test_dataset = dataset.stratified_split(dataset, test_size=0.15, valid_size=0.15)
    train_dataset.dataset.set_additional_transforms([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=45),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    ])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    trainer = ModelTrainer(device, model)
    trainer.setup(train_loader, valid_loader, criterion, optimizer)
    trainer.model = trainer.model.to(trainer.device)
    
    start_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    trainer.fit(NUM_OF_EPOCHS, start_time, True, MODELS_PATH)
    trainer.test(test_loader, CLASSES_LIST, start_time, MODELS_PATH)
    
def plot_random_10_images_with_predictions(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualNetwork(RES_BLOCKS_PER_LAYER, len(CLASSES_LIST))
    
    dataset = ClassificationDataset(dataset_root_dir=DATASET_PATH, dataset_name=DATASET_NAME, 
                                    dataset_url=DATASET_URL, classes_labels=CLASSES_LIST)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    classes = {i: None for i in range(10)}
    chosen_images = []
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                label = labels[i].item()
                if not classes[label]:
                    classes[label] = (images[i].cpu(), predicted[i].item(), label)
                if all(classes.values()): 
                    break
            if all(value is not None for value in classes.values()):
                break  

    chosen_images = [value for value in classes.values() if value is not None]

    _, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for ax, (image, pred, true) in zip(axs, chosen_images):
        ax.imshow(image.permute(1, 2, 0)) 
        ax.set_title(f"Expected: {CLASSES_LIST[true]}\nPredicted: {CLASSES_LIST[pred]}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_PATH, "examples.png"))
    plt.show()

if __name__ == "__main__":
    train_model()
    # plot_random_10_images_with_predictions("models/model_demo.pth")
