import matplotlib.pyplot as plt
import numpy as np
from FileLoader import *
from Preprocessing import *
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from Net import *
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import operator
from functools import reduce




np.set_printoptions(threshold = 1e6)

def add_paddings(list, max_corpus_length, model):
    temp = [model.wv[word] for word in list]
    ndarray = np.array(temp)

    if (max_corpus_length - len(list))%2 == 0:
        ndarray = np.pad(ndarray,((int((max_corpus_length - len(list))/2),int((max_corpus_length - len(list))/2)),(0,0)),'constant',constant_values = (0,0))
    else:
        ndarray = np.pad(ndarray, ((int((max_corpus_length - len(list)) / 2), int((max_corpus_length - len(list)) / 2)+1), (0, 0)),'constant', constant_values=(0, 0))
    return ndarray


file = FileLoader("Womens Clothing E-Commerce Reviews.csv")
data = file.read_file()
print(data.info())

data.drop(labels=['Clothing ID','Title'],axis=1,inplace = True)
data = data[~data['Review Text'].isnull()]


# ros = RandomOverSampler(random_state=0)
# data_resampled, label_resampled = ros.fit_resample(pd.DataFrame(data['Review Text']), data["Recommended IND"])
# duplicate = data[data["Recommended IND"].isin([0])]
# print(duplicate)
# data = pd.concat([data,duplicate,duplicate])
print(data)

preprocessed_data = Preprocessing(data)
preprocessed_data.error_cleaning("Review Text")

review_text = preprocessed_data.sentence_normalizatio("Review Text")



# Set values for various parameters
feature_size = 100    # Word vector dimensionality, i.e., number of dimension we wish to represent the word
window_context = 5  # Context window size
min_word_count = 1   # Minimum word count, i.e., words > this are going to be included in the model
sample = 1e-3        # Downsample setting for frequent words

corpus = [sentence for sentence in review_text]
w2v_model = word2vec.Word2Vec([sentence for sentence in corpus], size=feature_size,
                          window=window_context, min_count = min_word_count,
                          sample=sample)
print(corpus[0])
print(review_text[0])


temp_list = []
for i in range(len(corpus)):
    temp_list.append(corpus[i])
print(max(len(x) for x in temp_list))

max_corpus_length = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_list = []
for i in range(len(corpus)):
    data_list.append(add_paddings(corpus[i], max_corpus_length, w2v_model))
print(np.array(data_list).shape)

tensor_data = torch.tensor(np.array(data_list), device=device, dtype=torch.int64)
print(tensor_data.shape,tensor_data.shape[0],tensor_data.shape[1],tensor_data.shape[2])
tensor_data = torch.reshape(tensor_data,(tensor_data.shape[0],1,tensor_data.shape[1],tensor_data.shape[2]))
print(tensor_data.shape)
# tensor_data = tensor_data.to(dtype=torch.long)


tensor_label = torch.tensor(np.array(data["Recommended IND"]), device=device, dtype=torch.int64)
dataset = torch.utils.data.TensorDataset(tensor_data,tensor_label)
train_data,test_data=random_split(dataset,[round(0.8*tensor_data.shape[0]),round(0.2*tensor_data.shape[0])],generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=100,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=True)

net = Net()
net = net.to(device)
# weights = [1.0, 0.225]
# class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs_train, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs_train.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# dataiter = iter(test_loader)
# inputs_test, labels = dataiter.next()
#
# outputs = net(inputs_test.float())
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs_test, labels = data
        outputs = net(inputs_test.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in test_loader:
        inputs_test, labels = data
        outputs = net(inputs_test.float())
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = (0, 1)
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))