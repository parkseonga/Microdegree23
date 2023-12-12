import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pickle, os, warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from random import randint

# import cv2
from torch.autograd import Function
from scipy.interpolate import interp1d

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}") 

# Prespecifications

task = 'regression' # either 'classification' or 'regression'
invasive = False # either True or False
multi = False # either True or False, 아직 True 지원 안됨..
pred_lag = 300 # 300 for 5-min, 600 for 10-min, 900 for 15-min prediction or others

cuda_number = 0 # -1 for multi GPU support
num_workers = 0
batch_size = 256
max_epoch = 200

random_key = randint(0, 100000) # Prespecify seed number if needed

dr_classification = 0.6 # Drop out ratio for classification model
dr_regression = 0.0 # Drop out ratio for regression model

csv_dir = './capstone/data/model/'+str(random_key)+'/csv/'
pt_dir = './capstone/data/model/'+str(random_key)+'/pt/'

if not ( os.path.isdir( csv_dir ) ):
    os.makedirs ( os.path.join ( csv_dir ) )
    
if not ( os.path.isdir( pt_dir ) ):
    os.makedirs ( os.path.join ( pt_dir ) )


# Establish dataset

class MyDataset(Dataset):
    def __init__(self, ppg, y):
        self.ppg = ppg
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        ppg = self.ppg[idx]
        y = self.y[idx]
        return ppg, y
    
# Uni-dimensional convolutional neural network

class Net(nn.Module):
    def __init__(self, task, invasive, multi):
    
        self.task, self.invasive, self.multi = task, invasive, multi
        super(Net, self).__init__()

        if self.task == 'classification':
            self.dr = dr_classification
            self.final = 2
        else:
            self.dr = dr_regression
            self.final = 1

        if self.multi == True:
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1
                        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(320, self.final),
            nn.Dropout(self.dr)
        )
        
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        
        x = x.view(x.shape[0], self.inc, -1)
    
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))

        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)
        
        return out



# Read dataset
processed_dir = './capstone/data/'
cases = os.listdir(processed_dir)

if task == "classification":
    prefix = "clf_"
else:
    prefix = "reg_"

caseids_train = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}caseids_train_vf.pkl", "rb"))
caseids_valid = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}caseids_valid_vf.pkl", "rb"))
caseids_test = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}caseids_test_vf.pkl", "rb"))
x_train = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}x_train_vf.pkl", "rb"))
x_valid = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}x_valid_vf.pkl", "rb"))
x_test = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}x_test_vf.pkl", "rb"))
y_train = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}y_train_vf.pkl", "rb"))
y_valid = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}y_valin_vf.pkl", "rb"))
y_test = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}y_test_vf.pkl", "rb"))
c_train = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}c_train_vf.pkl", "rb"))
c_valid = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}c_valid_vf.pkl", "rb"))
c_test = pickle.load(open(f"{os.getcwd()}/capstone/data/{prefix}c_test_vf.pkl", "rb"))

print("caseids train cnt: {}, caseids val cnt {}, caseids test cnt: {}".format(len(caseids_train), len(caseids_valid), len(caseids_test)))
print("samples train cnt: {}, samples val cnt: {}, samples test cnt: {}".format(len(x_train), len(x_valid), len(x_test)))

train_dataset = MyDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

val_dataset = MyDataset(torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

test_dataset = MyDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 

# Define loader and model

if task == 'classification':
    task_target = 'hypo'
    criterion = nn.BCELoss()
else:
    task_target = 'map'
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

print ( '\n===== Task: {}, Seed: {} =====\n'.format ( task, random_key ) )
print ( 'Invasive: {}\nMulti: {}\nPred lag: {}\n'.format ( invasive, multi, pred_lag ))

# Model development and validation

# torch.cuda.set_device(cuda_number)
model = Net( task = task, invasive = invasive, multi = multi )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epochs = max_epoch

best_loss, best_auc = 99999.99999, 0.0

# 훈련 루프
for epoch in range(num_epochs):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for x_train, y_train in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            x_train, y_train = x_train.to(device), y_train.to(device) 
            optimizer.zero_grad()
            outputs = model(x_train)
                    
            loss = criterion(y_train,outputs)
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=loss.item())

    all_probs = []
    all_labels = []
    # 검증
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(val_loader)):
            x_valid, y_valid = batch
            x_valid, y_valid = x_valid.to(device), y_valid.to(device) 
            
            optimizer.zero_grad()
            outputs = model(x_valid)
                    
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_valid.cpu().numpy())
            
    diff_y = [i-j for i, j in zip(all_labels, all_probs)]
    mae = np.mean(np.abs(diff_y))
    print("MAE Score:", mae)

    # auc_score = roc_auc_score(all_labels, all_probs)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation MAE: {mae}%')

    label_invasive = 'invasive' if invasive == True else 'noninvasive'
    label_multi = 'multi' if multi == True else 'mono'
    label_pred_lag = str ( int ( pred_lag / 60 ) ) + 'min'
    filename = task+'_'+label_invasive+'_'+label_multi+'_'+label_pred_lag
    torch.save(model.state_dict(),pt_dir+filename+'_epoch_{0:03d}.pt'.format(epoch+1) )
    
    # 모델을 평가 모드로 설정
    model.eval()

    # 예측 확률을 저장할 리스트
    test_probs = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x_test, y_test = batch
            x_test, y_test = x_test.to(device), y_test.to(device) 
            
            outputs = model(x_test)
            test_probs.extend(outputs.cpu().numpy())

    diff_y = [i-j for i, j in zip(y_test.cpu().numpy(), test_probs)]

    # 실제 레이블과 예측 확률을 사용하여 AUC 계산
    mae = np.mean(np.abs(diff_y))
    print(f'Test MAE: {mae}')

    # train, valid loss 저장 코드 추가 
    # 결과 저장