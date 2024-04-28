import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix, precision_score, \
                            recall_score, f1_score, ConfusionMatrixDisplay

class ModelTrainer:
    def __init__(self, device, model):
        self.device = device
        self.model = model.to(device)
        self.init_weights()

        self.train_loader = None
        self.valid_loader = None
        self.criterion = None
        self.optimizer = None

        self.metrics = {
            'train_loss': [],         'val_loss': [],
            'train_precision': [],    'val_precision': [],
            'train_recall': [],       'val_recall': [],
            'train_f1': [],           'val_f1': []
        }

    def setup(self, train_loader, valid_loader, criterion, optimizer):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
    
    def init_weights(self):
        self.model.apply(self.fill_weights)

    @staticmethod
    def fill_weights(module):
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            torch.nn.init.kaiming_uniform_(module.weight)

    def calculate_metrics(self, outputs, labels):
        preds = np.argmax(outputs, axis=1)
        
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        return precision, recall, f1

    def plot_metrics(self, start_time, show_plots=False, save_path=None):
        _, axs = plt.subplots(2, 2)

        axs[0, 0].plot(self.metrics['train_loss'], label='Train')
        axs[0, 0].plot(self.metrics['val_loss'], label='Validation')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlabel('Epoch')

        axs[0, 1].plot(self.metrics['train_precision'], label='Train')
        axs[0, 1].plot(self.metrics['val_precision'], label='Validation')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].set_xlabel('Epoch')

        axs[1, 0].plot(self.metrics['train_recall'], label='Train')
        axs[1, 0].plot(self.metrics['val_recall'], label='Validation')
        axs[1, 0].set_ylabel('Recall')
        axs[1, 0].set_xlabel('Epoch')

        axs[1, 1].plot(self.metrics['train_f1'], label='Train')
        axs[1, 1].plot(self.metrics['val_f1'], label='Validation')
        axs[1, 1].set_ylabel('F1 Score')
        axs[1, 1].set_xlabel('Epoch')

        for ax in axs.flat:
            ax.legend()
            ax.grid(True)
            epochs = len(self.metrics['train_loss'])  
            ax.set_xticks(range(0, epochs)) 
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(f'{save_path}/metrics_{start_time}.png')
        if show_plots:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, preds, labels, classes, start_time, phase, save_path=None):
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix')
        
        if save_path is not None:
            plt.savefig(f'{save_path}/cm_{phase}_{start_time}.png')
        plt.close()

    def epoch_end(self, phase, epoch, start_time, outputs_list, labels_list, loss, save_path=None, show_plots=False):
        outputs = np.vstack(outputs_list)
        labels = np.concatenate(labels_list)
        
        precision, recall, f1 = self.calculate_metrics(outputs, labels)
        self.metrics[f'{phase}_loss'].append(loss)
        self.metrics[f'{phase}_precision'].append(precision)
        self.metrics[f'{phase}_recall'].append(recall)
        self.metrics[f'{phase}_f1'].append(f1)
        
        print(f'{phase.upper()} Epoch: {epoch}, Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        if phase == 'val':
            self.plot_metrics(start_time, show_plots=show_plots, save_path=save_path)
            preds = np.argmax(outputs, axis=1)
            self.plot_confusion_matrix(preds, labels, classes=np.arange(10), start_time=start_time, phase=phase, save_path=save_path)

    def train_one_epoch(self, epoch, start_time, show_plots=False, save_path=None):
        self.model.train()
        running_loss = 0.0
        outputs_list, labels_list = [], []

        for batch in self.train_loader:
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            outputs_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.epoch_end('train', epoch, start_time, outputs_list, labels_list, epoch_loss, save_path, show_plots)

    def validate_one_epoch(self, epoch, start_time, show_plots=False, save_path=None):
        self.model.eval()
        running_loss = 0.0
        outputs_list, labels_list = [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device, dtype=torch.long)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.valid_loader.dataset)
        self.epoch_end('val', epoch, start_time, outputs_list, labels_list, epoch_loss, save_path, show_plots)

    def fit(self, epochs, start_time, show_plots=None, save_path=None):
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            self.train_one_epoch(epoch, start_time, show_plots, save_path)
            clear_output(wait=True)
            self.validate_one_epoch(epoch, start_time, show_plots, save_path)
            if save_path is not None:
                torch.save(self.model.state_dict(), f'{save_path}/model_{start_time}.pth')

    def test(self, test_loader, classes, start_time, save_path=None):
        self.model.eval()
        test_loss = 0.0
        outputs_list, labels_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device, dtype=torch.long)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        test_loss /= len(test_loader.dataset)

        preds = np.concatenate([np.argmax(x, axis=1) for x in outputs_list], axis=0)
        labels = np.concatenate(labels_list, axis=0)
        precision, recall, f1 = self.calculate_metrics(np.vstack(outputs_list), np.concatenate(labels_list))
        
        print(f'Test Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        self.plot_confusion_matrix(preds, labels, classes=classes, start_time=start_time, phase='test', save_path=save_path)