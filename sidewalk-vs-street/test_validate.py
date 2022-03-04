import torch
from torch.utils.data import DataLoader, Dataset
from dataloader import SidewalkDataSet
from torchmetrics import ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from cnn_model import SidewalkClassifier
from tqdm import tqdm


def validate(model, dataset):
    reps = 10
    all_preds = []
    all_targets = []
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    accuracies = []
    for i in range(reps):
        predictions = []
        targets = []
        print('rep', i)
        for sample, target in tqdm(dataloader):
            preds = model(sample)
            predictions.extend(torch.argmax(preds, dim=0).tolist())
            targets.extend(torch.argmax(target, dim=0).tolist())
        all_preds.extend(predictions)
        all_targets.extend(targets)
        accuracy = sum(predictions == targets)/len(predictions)*100
        accuracies.append(accuracy)
    accuracy = sum(accuracies)/len(accuracies)
    matrix = confusion_matrix(all_targets, all_preds)
    conf = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['street', 'sidewalk'])

def val(trainer):
    results = []
    for i in range(10):
        res = trainer.validate



        
if __name__ == "__main__":
    val = "IMU_Streams/val_samples"
    constants = "IMU_Streams/data_stats_train.csv"
    dataset = SidewalkDataSet(val, constants)
    model = SidewalkClassifier(train_path='none', val_path=val, path_to_constants=constants)
