#!/home/user/miniconda3/envs/torch-gpu/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
import neptune.new as neptune
from sklearn.model_selection import KFold

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
print(len(train_embeddings))
test_embeddings = load_tensor(dir_input + "test_embeddings", torch.FloatTensor)
sequences = train_embeddings + test_embeddings
labels = label_train + label_test
print(len(labels))
print(len(sequences))
# shuffle data
dataset = list(zip(sequences, labels))
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
        #x = x.view(-1, x.size(0))
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = self.activation(self.fc4(x))
        return x

learning_rate = 0.00001
num_epochs = 10
k_folds = 5

# For fold results
results = {}
  
# Set fixed random number seed
torch.manual_seed(42)
# initialize network
model = ProteinClassifier(n_classes).to(device)
print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)
    
# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=10, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=10, sampler=test_subsampler)
    # Train network

    for epoch in range(0, num_epochs):

      # Print epoch
      print(f'Starting epoch {epoch+1}')

      # Set current loss value
      current_loss = 0.0

      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        #print(targets.shape)
        #print(targets.squeeze(-1).shape)
        #print(inputs.shape)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets.squeeze(-1))
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    """
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)
    """
    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data

        # Generate outputs
        outputs = model(inputs)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.squeeze(-1).size(0)
        correct += (predicted == targets.squeeze(-1)).sum().item()

      # Print accuracy
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
      print('--------------------------------')
      results[fold] = 100.0 * (correct / total)
    
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
    print(f'Average: {sum/len(results.items())} %')