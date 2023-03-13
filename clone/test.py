import pandas as pd
import torch
import time
import numpy as np
import warnings
import load_model
from gensim.models.word2vec import Word2Vec

from clone.classify_model import ClassifyModel
from clone.ensemble_model import EnsembleModel
from clone.model import BatchProgramClassifier
from model_1 import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


root = 'data/'
train_data = pd.read_pickle(root+'c/train/blocks.pkl').sample(frac=1)
test_data = pd.read_pickle(root+'c/test/blocks.pkl').sample(frac=1)

######################


root = 'data/'
word2vec = Word2Vec.load(root + "c/train/embedding/node_w2v_128").wv
MAX_TOKENS = word2vec.vectors.shape[0]
EMBEDDING_DIM = word2vec.vectors.shape[1]
embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
#设置参数
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 1
BATCH_SIZE = 32
USE_GPU = True

model1 = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                       USE_GPU, embeddings)
model2 = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                       USE_GPU, embeddings)
LABELS_classify = 104
BATCH_SIZE_classify = 64
USE_GPU_classify = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classify = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS_classify,
                                            BATCH_SIZE_classify,
                                            USE_GPU_classify, embeddings)
model_classify.load_state_dict(torch.load("best_model.pt", map_location=device))
ensem_classify = ClassifyModel(model_classify,model_classify)
if USE_GPU:
    model1.cuda()
    model2.cuda()

parameters = model1.parameters()
parameters = model2.parameters()
model1.load_state_dict(torch.load("clone_model2.pt"))
model2.load_state_dict(torch.load("clone_model1.pt"))
######################
BATCH_SIZE = 32
USE_GPU = True

loss_function = torch.nn.BCELoss()

print(train_data)
precision, recall, f1 = 0, 0, 0
# testing procedure
predicts = []
trues = []
total_loss = 0.0
total = 0.0
i = 0
start = time.time()
weights = [[0.5,0.3,0.2],[0.7,0.2,0.1],[0.2,0.2,0.6],[0.3,0.4,0.3],[0.4,0.2,0.4]]
ensemble_model = EnsembleModel(model1, model2, ensem_classify, weights[2])
for j in range(5):
    print(j)
    i = 0
    predicts = []
    trues = []
    ensemble_model = EnsembleModel(model1, model2, ensem_classify, weights[j])
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test1_inputs, test2_inputs, test_labels = batch
        if USE_GPU:
            test_labels = test_labels.cuda()

        #predicted = load_model.ensemble(model1, model2, model_classify, test1_inputs, test2_inputs, weights[j])
        predicted = load_model.get_clone_predicted(test1_inputs, test2_inputs, model1)
        # predicted = ensemble_model(test1_inputs, test2_inputs)
        predicted = np.where(predicted >= 0.5, 1, 0)
        #predicted = predicted.cpu().numpy()
        # print(predicted)
        predicts.extend(predicted)

        # print(predicted)
        # print(type(predicted))
        trues.extend(test_labels.cpu().numpy())
        total += len(test_labels)
        # total_loss += loss.item() * len(test_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
    print(time.time()-start)