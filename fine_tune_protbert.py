#!/home/user/miniconda3/envs/torch-gpu/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
import neptune.new as neptune


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + ".npy", allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# Load preprocessed data
sequences = []
dir_input = "protbert/"
label_train = load_tensor(dir_input + "labels_train", torch.LongTensor)
label_test = load_tensor(dir_input + "labels_test", torch.LongTensor)
train_embeddings = load_tensor(dir_input + "train_embeddings", torch.FloatTensor)
test_embeddings = load_tensor(dir_input + "test_embeddings", torch.FloatTensor)
# shuffle data
train_dataset = list(zip(train_embeddings, label_train))
test_dataset = list(zip(train_embeddings, label_test))
n_classes = 2


# model architecture


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 100)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(32, n_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, x.size(0))
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = self.activation(self.fc4(x))
        return x


learning_rate = 0.00001
num_epochs = 10
# initialize network
model = ProteinClassifier(n_classes).to(device)
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
        x_train = x_train.to(device=device) #(1024)
        y_train = y_train.to(device=device)
        print("shape:", x_train.shape)
        x_train = x_train.reshape(x_train.shape[0], 1) #(1024, 1)
        print(x_train.shape)
        # forward
        scores = model(x_train)
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
        x_val = x_val.reshape(x_val.shape[0], 1)
        scores = model(x_val)
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
        x_test = x_test.reshape(x_test.shape[0], 1)
        # forward pass
        scores = model(x_test)
        # validation batch loss
        loss = criterion(scores, y_test)
        # calculate the accuracy
        _, predictions = scores.max(1)
        correct_test += (predictions == y_test).sum().item()

## PRINT TEST RESULTS ##
test_loss /= len(test_dataset)
accuracy = correct_test / len(test_dataset)
print(f"Test loss: {test_loss}.. Test Accuracy: {accuracy}")

