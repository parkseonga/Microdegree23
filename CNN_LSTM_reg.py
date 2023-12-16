import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

# 모델 정의
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        # CNN 레이어
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=10, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3) 

        self.flatten = nn.Flatten()
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.flatten(x)
        x = x.transpose(1, 2)  # LSTM에 맞게 차원 변경
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]  # LSTM의 마지막 hidden state
        x = self.fc(x)
        return x


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
    
def scale_data(data): # ppg scale
    rng = np.nanmax(data) - np.nanmin(data)
    minimum = np.nanmin(data)
    data = (data - minimum) / rng
    return data

def fill_2d_missing_with_previous(arr): # 결측값 이전 값으로 대체 
    # 결측값이 아닌 값들의 위치를 찾습니다.
    valid = np.isnan(arr) == False
    
    # 누적 최대값을 사용하여 각 위치에서 마지막으로 관찰된 유효한 값을 채웁니다.
    filled = np.maximum.accumulate(valid, axis=1)

    # np.where를 사용하여 유효한 값과 결측값의 위치에 따라 값을 선택합니다.
    return np.where(filled, arr, np.nan)

loaded_dataset = np.load('./all_reg_arr.npz', allow_pickle = True)
x_arr = loaded_dataset['x_arr']
y_arr = loaded_dataset['y_arr']
c_arr = loaded_dataset['c_arr']
a_arr = loaded_dataset['a_arr']

x_arr = x_arr[~np.isnan(y_arr)]
c_arr = c_arr[~np.isnan(y_arr)]
a_arr = a_arr[~np.isnan(y_arr)]
y_arr = y_arr[~np.isnan(y_arr)]

x_arr = x_arr[:,:,1].reshape(-1, 1, 3000)
x_arr = fill_2d_missing_with_previous(x_arr)

# x 에 nan 이 없는 것만 남기기 
idx = []
for i in range(len(x_arr)):
    if np.isnan(x_arr[i]).sum() == 0:
        idx.append(i)
        
x_arr = x_arr[idx].astype(float)
y_arr = y_arr[idx].astype(float)
c_arr = c_arr[idx]

# train, valid, test split (caseid 기준)
caseids = list(np.unique(c_arr))
nvalid = max(1, int(len(caseids) * 0.4))
ntest = max(1, int(len(caseids) * 0.2))

caseids_train = caseids[nvalid:]
caseids_valid = caseids[ntest:nvalid]
caseids_test = caseids[:ntest]

print("caseids train cnt: {}, caseids test cnt: {}".format(len(caseids_train), len(caseids_test)))

train_mask = np.isin(c_arr, caseids_train).flatten()
valid_mask = np.isin(c_arr, caseids_valid).flatten()
test_mask = np.isin(c_arr, caseids_test).flatten()

x_train = x_arr[train_mask]
y_train = y_arr[train_mask]
c_train = c_arr[train_mask]

x_valid = x_arr[valid_mask]
y_valid = y_arr[valid_mask]
c_valid = c_arr[valid_mask]

x_test = x_arr[test_mask]
y_test = y_arr[test_mask]
c_test = c_arr[test_mask]

for i in range(len(x_train)):
    x_train[i,0,:] = scale_data(x_train[i,0,:]) #, ppg_min, ppg_max)

for i in range(len(x_valid)):
    x_valid[i,0,:] = scale_data(x_valid[i,0,:]) #, ppg_min, ppg_max)

for i in range(len(x_test)):
    x_test[i,0,:] = scale_data(x_test[i,0,:]) #, ppg_min, ppg_max)

x_train = torch.tensor(x_train)
x_valid = torch.tensor(x_valid)

y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)

    
# 하이퍼파라미터 설정
learning_rate = 0.003
num_epochs = 50
batch_size = 1024

# torch 데이터 로드
train_dataset = MyDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) 

val_dataset = MyDataset(torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 

test_dataset = MyDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 

# set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 모델, 손실 함수, 최적화 알고리즘 초기화
model = CNNLSTM()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 학습률(lr)은 필요에 따라 조정
criterion = nn.L1Loss() 
model = model.to(device)

import matplotlib.pyplot as plt 
plt.figure(figsize = (7,7))
# 훈련 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_loss_values = []
    with tqdm(train_loader, unit="batch") as tepoch:
        for x_train, y_train in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            x_train, y_train = x_train.to(device), y_train.reshape(-1,1).to(device) 
            optimizer.zero_grad()
            outputs = model(x_train)
                    
            loss = criterion(y_train,outputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_values.append(train_loss)

            tepoch.set_postfix(loss=loss.item())

    all_probs = []
    all_labels = []
    # 검증
    model.eval()
    val_loss = 0.0
    val_loss_values = []
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(val_loader)):
            x_valid, y_valid = batch
            x_valid, y_valid = x_valid.to(device), y_valid.to(device) 
            
            optimizer.zero_grad()
            outputs = model(x_valid)
            
            loss = criterion(y_valid,outputs)            
            val_loss += loss.item()
            val_loss_values.append(val_loss)

            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_valid.cpu().numpy())
      
    plt.plot(np.array(train_loss_values), 'r', label = 'train loss')
    plt.plot(np.array(val_loss_values), 'r', label = 'val loss')
    plt.legend()
    plt.savefig('loss plot.png', bbox_inches = 'tight')
    
    diff_y = [i-j for i, j in zip(all_labels, all_probs)]
    mae = np.mean(np.abs(diff_y))
    print("MAE Score:", mae)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation MAE: {mae}%')

# 모델을 평가 모드로 설정
model.eval()

  # 예측 확률을 저장할 리스트
  test_probs = []
  y_tests = []
  with torch.no_grad():
      for i, batch in enumerate(tqdm(test_loader)):
          x_test, y_test = batch
          x_test, y_test = x_test.to(device), y_test.to(device) 
          
          outputs = model(x_test)
          y_tests.extend(y_test.cpu().numpy())
          test_probs.extend(outputs.cpu().numpy())

diff_y = [i-j for i, j in zip(y_tests.cpu().numpy(), test_probs)]

# 실제 레이블과 예측 확률을 사용하여 AUC 계산
mae = np.mean(np.abs(diff_y))
print(f'Test MAE: {mae}')

import pickle 
pickle.dump((y_tests, test_probs), open('./preds.pkl', 'wb'))
