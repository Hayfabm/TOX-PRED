#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
import numpy as np
import pandas as pd
import os
from utils import create_dataset, PC_6

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
TRAIN_SET = "~/TOX-pred/Dataset/7996_dataset/train_set.csv"
TEST_SET = "~/TOX-pred/Dataset/7996_dataset/test_set.csv"


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# create train dataset
sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)
labels_train = pd.DataFrame(labels_train)
label_train = torch.tensor(np.array(labels_train), dtype=torch.long)

# create test dataset
sequences_test, labels_test = create_dataset(data_path=TEST_SET)
labels_test = pd.DataFrame(labels_test)
label_test = torch.tensor(np.array(labels_test), dtype=torch.long)


# generate PC6-protein-encoding-method

pc6_train = PC_6(sequences_train, length=1000)
pc6_train = torch.tensor(
    np.array(list(pc6_train[: len(sequences_train)])), dtype=torch.float
)
print(pc6_train.shape)
pc6_test = PC_6(sequences_test, length=1000)
pc6_test = torch.tensor(
    np.array(list(pc6_test[: len(sequences_test)])), dtype=torch.float
)
print(pc6_test.shape)

# shuffle data
train_dataset = list(zip(pc6_train, label_train))
test_dataset = list(zip(pc6_test, label_test))
# model architecture


class CNN(nn.Module):
    def __init__(self, n_classes):  # input_shape(m, 1000,6)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 3), stride=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.dropout2 = nn.Dropout(0.25)
        self.fc = nn.Linear(8032, 100)
        self.fc1 = nn.Linear(100, n_classes)

    def forward(self, x):
        # print("first_x", x.shape)
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        # print("x_shape:", x.shape)
        # x = self.bn1(x)
        # print("x_shape:", x.shape)
        x = self.maxpool1(x)
        # print("x_shape:", x.shape)
        # x = self.dropout1(x)
        # print("x_shape:", x.shape)
        x = F.relu(self.conv2(x))
        # print("x_shape:", x.shape)
        # x = self.bn2(x)
        # print("x_shape:", x.shape)
        x = self.maxpool2(x)
        # print("x_shape:", x.shape)
        # x = self.dropout2(x)
        # print("x_shape:", x.shape)
        x = torch.flatten(x)
        # print("x_shape:", x.shape)
        x = F.relu(self.fc(x))
        # print("x_shape:", x.shape)
        x = torch.sigmoid(self.fc1(x))
        return x


learning_rate = 0.00001
num_epochs = 100
n_classes = 2
# initialize network
model = CNN(n_classes).to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):

    ### TRAIN LOOP ###
    # Print
    print("epochs:", epoch)

    # Print
    train_loss = 0.0
    correct_train = 0
    total = len(train_dataset)
    for i, (x_train, y_train) in enumerate(train_dataset, 1):
        # Get data to cuda if possible
        x_train = x_train.to(device=device)
        y_train = y_train.to(device=device)
        # print("***********", x_train.shape)

        # forward
        scores = model(x_train).unsqueeze(0)

        # print("s", scores.shape)
        # print(("y", y_train.shape))
        loss = criterion(scores, y_train)

        _, predictions = scores.max(1)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam strep
        optimizer.step()

        # accuracy & loss
        train_loss += loss.item()
        correct_train += predictions.eq(y_train).sum().item()

        train_epoch_loss = train_loss / total
        train_epoch_acc = 100 * (correct_train / total)
    print(
        "training_loss: {:.2f} | training_acc: {:.2f}".format(
            train_epoch_loss, train_epoch_acc
        )
    )
    print("-" * 70)

"""
### VALIDATION LOOP ###

# set the model to eval mode
model.eval()
valid_loss = 0
total = len(val_dataset)
print("evaluting trained model ...")
y_pred = []
with torch.no_grad():
    for x_val, y_val in val_dataset:
        x_val = x_val.to(device=device)
        y_val = y_val.to(device=device)
        scores = model(x_val).unsqueeze(0)
        # validation batch loss
        loss = criterion(scores, y_val)
        # accumulate the valid_loss
        valid_loss += loss.item()

## PRINT EPOCH RESULTS ##
train_loss /= len(train_dataset)
valid_loss /= len(val_dataset)
print(
    f"Epoch: {epoch+1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}"
)
"""
### TEST LOOP ###
# set the model to eval mode
model.eval()
test_loss = 0
correct_test = 0
# turn off gradients for validation
with torch.no_grad():
    for x_test, y_test in test_dataset:
        x_test = x_test.to(device=device)
        y_test = y_test.to(device=device)

        # forward pass
        scores = model(x_test).unsqueeze(0)

        # validation batch loss
        loss = criterion(scores, y_test)
        # calculate the accuracy
        _, predictions = scores.max(1)
        correct_test += (predictions == y_test).sum().item()

## PRINT TEST RESULTS ##
test_loss /= len(test_dataset)
accuracy = correct_test / len(test_dataset)
print(f"Test loss: {test_loss}.. Test Accuracy: {accuracy}")
