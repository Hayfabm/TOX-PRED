"""utils file"""
from typing import List, Tuple
import os
import numpy as np
import pandas as pd
from utils import create_dataset
from biotransformers import BioTransformers

# training parameters
TRAIN_SET = "/home/haifa/TOX-pred/Dataset/7996_dataset/train_set.csv"
TEST_SET = "/home/haifa/TOX-pred/Dataset/7996_dataset/test_set.csv"
BIOTF_MODEL = "protbert"
BIOTF_POOLMODE = "cls"
BIOTF_BS = 2


# create train dataset
sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)
labels_train = pd.DataFrame(labels_train)
label_train = np.array(labels_train)
print(label_train.shape)
# create test dataset
sequences_test, labels_test = create_dataset(data_path=TEST_SET)
labels_test = pd.DataFrame(labels_test)
label_test = np.array(labels_test)
print(label_test.shape)

# generate embeddings
bio_trans = BioTransformers(backend=BIOTF_MODEL)

sequences_train_embeddings = bio_trans.compute_embeddings(
    sequences_train[1:5], pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
)[BIOTF_POOLMODE]
print(sequences_train_embeddings)
print(sequences_train_embeddings.shape)
sequences_test_embeddings = bio_trans.compute_embeddings(
    sequences_test[1:5], pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
)[BIOTF_POOLMODE]

# list to Tensor
"""
dir_input = "protbert/"
os.makedirs(dir_input, exist_ok=True)

np.save(dir_input + "labels_train", label_train)
np.save(dir_input + "labels_test", labels_test)
np.save(dir_input + "train_embeddings", sequences_train_embeddings)
np.save(dir_input + "test_embeddings", sequences_test_embeddings)

"""
