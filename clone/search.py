'''
1.读取.c和.cpp文件
2.把文件内容转化成字符串的格式
3.调用load_model里的trans_seq函数，把字符串转化成输入模型的格式
4.读取所有数据
all_data = pd.read_pickle('data/c/train/blocks.pkl').sample(frac=1)
5.将待搜索的和all_data里所有的依次放入模型
output = load_model.get_classify_predicted(test1_inputs, test2_inputs)
predicted = (output.data > 0.5).cpu().numpy()
6.记录下predicted输出为1的位置对于的id号，输出id号并写入文件
'''
import time

import numpy as np
import pandas as pd
import torch
from ensemble_model import EnsembleModel
from classify_model import ClassifyModel
from gensim.models import Word2Vec

import load_model
from model import BatchProgramClassifier
from model_1 import BatchProgramCC


def search(file_content, dl, index,queue):
    root = 'data/'
    word2vec = Word2Vec.load(root + "c/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
    # 设置参数
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    BATCH_SIZE = 32
    USE_GPU = False

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
    ensem_classify = ClassifyModel(model_classify, model_classify)
    if USE_GPU:
        model1.cuda()
        model2.cuda()

    model1.load_state_dict(torch.load("clone_model2.pt",map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load("clone_model1.pt",map_location=torch.device('cpu')))
    ######################
    weights = [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3], [0.4, 0.2, 0.4]]
    ensemble_model = EnsembleModel(model1, model2, ensem_classify, weights[0])

    # 读取输入的文件并转换格式
    batch_size = 24

    input_data = load_model.trans_seq(file_content)
    input_sin_data = input_data.tolist()[0]
    input_mul_data = []
    for i in range(batch_size):
        input_mul_data.append(input_sin_data)

    # 加载库中所有数据
    start = time.time()
    all_data = pd.read_pickle('all_code_data.pkl').sample(frac=1)
    data_start = int(len(all_data) * dl * index)
    data_end = int(data_start + len(all_data) * dl)
    all_data = all_data.iloc[data_start:data_end]
    print(str(data_start)+" "+str(data_end))

    i = 0
    kv = {}
    start = time.time()

    # 增量搜索
    while i < len(all_data):
        batch = load_model.get_batch(all_data, i, batch_size)
        i += batch_size
        id, test_inputs, test_labels = batch
        if False:
            test_labels = test_labels.cuda()

        # output = load_model.get_clone_predicted(input_mul_data[0:len(test_inputs)], test_inputs, model_clone_1)
        output = ensemble_model(input_mul_data[0:len(test_inputs)],
                                     test_inputs)
        # predicted = (output.data > 0.5).cpu().numpy()
        # 添加检索到的数据
        true_indices = np.where(output == True)[0]
        for index in true_indices.tolist():
            kv.update({str(id[index]):load_model.similarity(input_data,pd.Series([pd.Series(test_inputs[index]).tolist()]))})

        if(len(kv)>=10):
            break
        if i % 320 == 0:
            print(i/(time.time() - start))
            '''l = len(result)
            print(l / (time.time() - start))'''

    # for s,r in sim,result:
    #     print(str(s)+" "+str(r))
    print(time.time() - start)
    queue.put(kv)