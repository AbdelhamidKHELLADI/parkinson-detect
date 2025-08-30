import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NeuralNet(nn.Module):
    def __init__(self, input_size=23, num_classes=2):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 23)
        self.last= nn.Linear(23,num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.last(out)
        return out
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data ,targets in train_loader:
            data, targets = data.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
def evaluate_model(model, test_loader,num_classes=2):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, targets in test_loader:
            data= data.to(device)
            outputs = model(data)
            if num_classes == 2:
                preds= torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds= torch.sigmoid(outputs).cpu().numpy() > 0.5
                preds=preds.astype(int).squeeze()
            all_predictions.extend(preds)
            all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_predictions)

def load_data(path):
    df=pd.read_csv(path)
    df["label"]=df["path"].apply(lambda x: x.split("/")[-1].split(".")[0].lower())
    
    label_mapping={"healthy":0,"mild":1,"severe":2}
    df["label"]=df["label"].map(label_mapping)
    feature_columns=[col for col in df.columns if col not in ["path","audio_id","segment","label"]]
    X=df[feature_columns].values.astype(np.float32)
    y=df["label"].values.astype(np.float32)
    groups=df["audio_id"].values

    gkf = GroupKFold(n_splits=3)
    accuracies, precisions, recalls, f1s = [], [], [], []
    splits = list(gkf.split(X, y, groups))            # materialize splits so tqdm can infer length
    pbar = tqdm(enumerate(splits, 1), total=len(splits), desc="Training folds")
    for fold, (train_idx, val_idx) in pbar:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset=TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset=TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader=DataLoader(val_dataset, batch_size=64)

        model=NeuralNet(input_size=X.shape[1], num_classes=len(label_mapping)).to(device)
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.001)

        # train once for 100 epochs
        train_model(model, train_loader, criterion, optimizer, num_epochs=100)

        y_true, y_pred = evaluate_model(model, val_loader,num_classes=len(label_mapping))
        acc=accuracy_score(y_true, y_pred)
        prec=precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec=recall_score(y_true, y_pred, average='weighted')
        f1=f1_score(y_true, y_pred, average='weighted')
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        pbar.set_postfix({"Acc":f"{np.mean(accuracies):.4f}","Prec":f"{np.mean(precisions):.4f}","Rec":f"{np.mean(recalls):.4f}","F1":f"{np.mean(f1s):.4f}"})

    print(f"Final Results - Acc: {np.mean(accuracies):.4f}, Prec: {np.mean(precisions):.4f}, Rec: {np.mean(recalls):.4f}, F1: {np.mean(f1s):.4f}")
    return model

def main():
    data_path="/media/data/features/extracted_features.csv"
    model=load_data(data_path)
    torch.save(model.state_dict(), "./models/parkinson_3classes_model.pth")
    print("Model saved to ./models/parkinson_3classes_model.pth")
if __name__ == "__main__":
    main()