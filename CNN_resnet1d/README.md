## resnet-1d

### read data files

#### [ver1] 1 trial for noise processing

- get rid of nan value in each arr segment
  - x
    - if all values of segment is nan, change nan -> 0
    - if some values of segment is nan, change nan -> mean of prior values
  - y
    - change nan -> 2 (for classification)

#### [final] final version of noise processing

1. if there is nan value in x, y segment, exclude from dataset
2. if there is value below 0 in x segment, exclude from dataset
3. if number of peak in x segment below 10, exclude from dataset
4. if beat in x segment is irregular(based on average of beat length), exclude from dataset
5. if correlation of beat in x segment is less than 0.9, exclude from dataset

### train

- train_clf file for classification task
  - loss : nn.CrossEntropyLoss, nn.BCEWithLogitsLoss
- train_reg file for regression task
  - loss : nn.MSELoss, nn.L1Loss

### model
- reference : https://github.com/hsd1503/resnet1d
