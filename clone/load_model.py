import csv

from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_recall_fscore_support

from model import BatchProgramClassifier
import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model_1 import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
import sys
from tqdm.auto import tqdm

tqdm.pandas()
warnings.filterwarnings('ignore')


#读取数据集，返回id和对应的代码
def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    id, data, labels = [], [], []
    for _, item in tmp.iterrows():
        id.append(item[0])
        data.append(item['code'])
        labels.append(item[2]-1)
    return id, data, torch.LongTensor(labels)


def get_clone_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    id1,x1, x2, labels = [], [], [],[]
    for _, item in tmp.iterrows():
        id1.append(item["id1"])
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return id1 , x1, x2, torch.FloatTensor(labels)

def get_clone_predicted_by_classify(input1,input2,model_classify):
    classify1 = get_classify_predicted(input1,model_classify)
    classify2 = get_classify_predicted(input2,model_classify)
    equal_mask = torch.eq(classify1, classify2)

    # 获取相等位置的下标
    equal_indices = equal_mask.nonzero()

    # 将相等位置的值设置为True，其他位置的值设置为False
    result = torch.zeros(len(classify1), dtype=bool)
    result[equal_indices] = True
    result = result.unsqueeze(dim=1).numpy()
    return result

#获取分类结果
def get_classify_predicted(input,model_classify):
    model_classify.batch_size = len(input)
    model_classify.hidden = model_classify.init_hidden()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    output = model_classify(input)
    _, predicted = torch.max(output.data, 1)
    return predicted
def get_clone_predicted(input1,input2,model_clone_1):
    model_clone_1.batch_size = len(input1)
    model_clone_1.hidden = model_clone_1.init_hidden()
    output = model_clone_1(input1, input2)
    predicted = (output.data > 0.5).cpu().numpy()
    return predicted

import numpy as np
from scipy.spatial.distance import cosine
def similarity(code1,code2):
    vec1 = np.array([get_vector(seq) for seq in code1])
    vec2 = np.array([get_vector(seq) for seq in code2])
    similarity_matrix = cosine_similarity(vec1, vec2)
    similarity_score = similarity_matrix[0][0]  # Assuming we're comparing only two code snippets
    if similarity_score>=0.98:
        return "high"
    else:
        return "low"

#将字符串转换成输入的格式
def trans_seq(str):
    from pycparser import c_parser
    parser = c_parser.CParser()
    str_series = pd.Series(str)
    str_parser = str_series.progress_apply(parser.parse)
    return generate_block_seqs(str_parser)


def generate_block_seqs(str_parser):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load('data/c/train/embedding/node_w2v_' + str(128)).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        #trees = pd.read_pickle(data_path)
        #trees['code'] = trees['code'].apply(trans2seq)
        str_trans2seq = str_parser.apply(trans2seq)
        #str_trans2seq.to_csv("str_trans2seq.csv")
        return str_trans2seq

#找到id对应的代码
def find_code_by_id(id_list, csv_file):
    kv = {}
    # 读取 csv 文件
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # 跳过文件的第一行，因为第一行是表头
        next(reader)
        # 遍历每一行
        for row in reader:
            # 如果当前行的 id 在输入的列表中，则返回该行的 code 数据
            if row[0] in id_list:
                kv.update({row[0]:row[1]})
    # 如果遍历完整个 csv 文件都没有找到对应的 id，则返回 None
    return kv


def get_vector(seq):
    # Convert a sequence of syntax blocks into a fixed-length vector
    word2vec = Word2Vec.load('data/c/train/embedding/node_w2v_128').wv
    max_token = word2vec.vectors.shape[0]

    vector = np.zeros((128,))
    count = 0
    for block in seq:
        for token in block:
            if isinstance(token, int) and token < max_token:
                vector += word2vec.vectors[token]
                count += 1
    if count > 0:
        vector /= count
    return vector


def ensemble(model_clone_1,model_clone_2,model_classify,input1, input2, weights=[0.4,0.2,0.4]):
    '''
    classifier1_output: 第一个分类器的预测结果，为Numpy数组
    classifier2_output: 第二个分类器的预测结果，为Numpy数组
    weights: 两个分类器的权重系数，为一个长度为2的列表，满足所有权重系数之和等于1
    '''
    clone1_output = get_clone_predicted(input1, input2,model_clone_1)
    clone2_output = get_clone_predicted(input1, input2, model_clone_2)
    classifier_output = get_clone_predicted_by_classify(input1, input2, model_classify)

    # 将Numpy数组中的浮点数四舍五入为整数
    output1 = np.round(clone1_output).astype(int)
    output2 = np.round(clone2_output).astype(int)
    output3 = np.round(classifier_output).astype(int)

    # 计算加权平均结果
    ensemble_output = np.round((output1 * weights[0]) + (output2 * weights[1]) +
                               (output3 * weights[2])).astype(bool)

    # 将结果转换为numpy数组并返回
    return ensemble_output.reshape(-1, 1)

import pycparser
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
'''
content1 = ""
with open('example1.cpp', 'r') as f:
    content1 = f.read()
content2 = ""
with open('example3.cpp', 'r') as f:
    content2 = f.read()
input_data1 = trans_seq(content1)
input_data2 = trans_seq(content2)
input_data1_list = input_data1.tolist()[0]
input_mul_data1 = []
for i in range(32):
    input_mul_data1.append(input_data1_list)
print(input_mul_data1[0])
print(input_data1)
lst = pd.Series(input_mul_data1[0]).tolist()
print(pd.Series([pd.Series(input_mul_data1[0]).tolist()]))
print(type(input_data1))
input_data2 = input_data2.tolist()[0]
input_mul_data2 = []
for i in range(32):
    input_mul_data2.append(input_data2)
#print(type(input_data1))
print(get_clone_predicted_by_classify(input_mul_data1,input_mul_data2))'''
#print(type(get_clone_predicted(input_mul_data1,input_mul_data2)) )
'''print(similarity(pd.Series([pd.Series(input_mul_data1[0]).tolist()]),pd.Series([pd.Series(input_mul_data2[0]).tolist()])))
print((get_clone_predicted(input_data1,input_data2).data > 0.5).cpu().numpy())'''
# 加载词向量

