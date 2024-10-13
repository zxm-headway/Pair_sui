import os
import torch
import re
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import logging
import time
import random
import argparse
import logging

import utils.log_help as log

from sklearn.model_selection import StratifiedKFold
from utils.order_loss import *
from models.GCAP import GRU_CNN_Attention as models
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ordered_set import OrderedSet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import os
import torch
import re
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import logging
import time
import random
import argparse
import logging

import utils.log_help as log

from utils.order_loss import *
from models.GCAP import GRU_CNN_Attention as models
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ordered_set import OrderedSet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score



# 读取reddit文件数据
def read_reddit_data(file_path = 'data/reddit_clean.pkl'):
    # 读取pkl文件
    reddit_data = pd.read_pickle(file_path)
    label  = [ user['label'] for user in reddit_data]
    users = [ user['user'] for user in reddit_data]
    return reddit_data,label,users

# 创建reddit数据集的词汇表集合
def get_vocabulary(reddit_data):
    # 创建一个空集合
    vocabulary = OrderedSet()
    words_id = {}
    id_words = {}
    # 遍历reddit数据集
    for posts in reddit_data:
        for post in posts['subreddit']:
            # 将文本分割为单词
            words = post.strip().split()
            # 将单词添加到词汇表集合中
            for word in words:
                vocabulary.add(word)

    # 统计词频并创建词汇表
    # word_counts = Counter(word_set)
    vocab = ["<PAD>", "<UNK>"] + list(vocabulary)
    words_id = {word: idx for idx, word in enumerate(vocab)}
    id_words = {idx: word for word, idx in words_id.items()}
    print(f"Total {len(vocab)} words found in all txt files")

    return vocab,words_id,id_words



# 数值化在batch传递的时候进行转化

# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx,reddit_data):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.reddit_data = reddit_data

    
    # 读取每个样本的文本
    def read_text(self, user,reddit_data):
        user_index = re.findall(r'\d+', user)
        user_index = int(user_index[0])
        user_posts = reddit_data[user_index]['subreddit']
        return user_posts



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text = self.read_text(text,self.reddit_data)

        # 将词转换为索引
        text_indices = []
        for post in text:
            if len(post) == 0:
                post_indices = []
            else:
                post_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in post]
            text_indices.append(post_indices)

        # print(text_indices)
        return text_indices, label


# 自定义 collate_fn 函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    # 找到批次中最大的帖子数量和最大的词数量
    max_num_posts = max(len(text) for text in texts)
    # max_num_words = max((len(post) for text in texts for post in text), default=1)
    max_num_words = min(512, max((len(post) for text in texts for post in text), default=1))


    batch_size = len(texts)
    # 初始化填充后的张量和掩码
    padded_texts = torch.zeros(batch_size, max_num_posts, max_num_words, dtype=torch.long)

    # 创建掩码
    post_masks = torch.zeros(batch_size, max_num_posts, max_num_words, dtype=torch.bool)
    text_masks = torch.zeros(batch_size, max_num_posts, dtype=torch.bool)

    for i, text in enumerate(texts):
        num_posts = len(text)
        text_masks[i, :num_posts] = 1  # 标记实际存在的帖子位置
        for j, post in enumerate(text):
            # num_words = len(post)
            num_words = min(len(post), max_num_words)  # 限制单词数量不超过 256
            if num_words > 0:
                # 实际每个帖子的词
                padded_texts[i, j, :num_words] = torch.tensor(post[:num_words])
                post_masks[i, j, :num_words] = 1  # 标记实际存在的词位置



    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels, text_masks, post_masks





    
# 设置随机种子
def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True






def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing GRU CLASS Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--save_log", type=str, default='./log_record/')
    parser.add_argument("--save_model_path", default='../check_point/', type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--filter_sizes", type=int, default=[2,3,4])
    parser.add_argument("--cnn_size", type=int, default=32)
    parser.add_argument("--gru_size", type=int, default=64)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--class_num", default=5, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(args)




# 训练模型
def train(model, iterator, optimizer,  device,logs,args):
    model.train()
    epoch_loss = 0
    out_result = []
    label_result = []
    for batch in tqdm(iterator, desc="Training"):
        text, label, text_masks, post_masks = batch
        text = text.to(device)
        label = label.to(device)
        text_masks = text_masks.to(device)
        post_masks = post_masks.to(device)
        optimizer.zero_grad()
        output = model(text, text_masks, post_masks)
        loss = loss_function(output, label, loss_type='ce', expt_type=args.class_num, scale=2)
        _, predicted = torch.max(output.data, 1)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        label_result.extend(label.cpu().detach().numpy().tolist())
        out_result.extend(predicted.cpu().detach().numpy().tolist())

    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f'Train Loss: {epoch_loss/len(iterator):.5f} | Train GP: {GP:.5f} | Train GR: {GR:.5f} | Train FS: {FS:.5f}')
    
    return epoch_loss / len(iterator),GP


# 评估模型
def evaluate(model, iterator,  device,logs,args):
    model.eval()
    epoch_loss = 0
    out_result = []
    label_result = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text, label, text_masks, post_masks = batch
            labels.append(label)
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)
            output = model(text, text_masks, post_masks)
            loss = loss_function(output, label, loss_type='ce', expt_type=args.class_num, scale=2)
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            label_result.extend(label.cpu().numpy().tolist())
            out_result.extend(predicted.cpu().numpy().tolist())
            
    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f'Val Loss: {epoch_loss/len(iterator):.5f} | Val GP: {GP:.5f} | Val GR: {GR:.5f} | Val FS: {FS:.5f}')
    return GP,GR,FS

# 测试模型
def test(model, iterator, device,logs):
    model.eval()
    out_result = []
    label_result = []
    with torch.no_grad():
        for batch in iterator:
            text, label, text_masks, post_masks = batch
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)
            output = model(text, text_masks, post_masks)
            _, predicted = torch.max(output.data, 1)
            label_result.extend(label.cpu().numpy().tolist())
            out_result.extend(predicted.cpu().numpy().tolist())

    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f' test GP: {GP:.5f} | test GR: {GR:.5f} | test FS: {FS:.5f}')
    return GP,GR,FS



# 训练和评估模型（每一折交叉验证）
def train_and_evaluate_with_kfold(args, device, logs):
    # 创建5折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    reddit_data, labels, texts = read_reddit_data()
    vocabulary, words_id, _ = get_vocabulary(reddit_data)
    all_gp_scores = []
    all_gr_scores = []
    all_fs_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
        logging.info(f'Fold {fold+1}')

        # 获取训练集和验证集
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 创建数据集
        train_dataset = TextDataset(train_texts, train_labels, words_id, reddit_data)
        train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

        val_dataset = TextDataset(val_texts, val_labels, words_id, reddit_data)
        val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

        # 初始化模型
        model = models(args=args, vocab_size=len(vocabulary), device=device)
        model = model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 保存每折中的最佳F1分数和对应模型
        best_fs = 0  # 保存每折最佳F1分数
        best_model_path = f'./k_models/best_model_fold_{fold}.pth'  # 为每一折创建独立的模型文件路径

        # 训练和验证当前折
        for epoch in range(args.epochs):
            logging.info(f'Epoch: {epoch + 1:02}')
            train_loss, _ = train(model, train_iterator, optimizer, device, logs, args)

            # 每个epoch结束后进行验证
            GP, GR, FS = test(model, val_iterator, device, logs)

            # 如果当前模型的F1分数比之前的最佳分数好，则保存模型
            if FS > best_fs:
                best_fs = FS
                torch.save(model.state_dict(), best_model_path)  # 保存当前模型参数

        # 加载当前折的最佳模型
        logging.info(f"Loading best model from fold {fold+1} with F1 score: {best_fs}")
        best_model = models(args=args, vocab_size=len(vocabulary), device=device)
        best_model.load_state_dict(torch.load(best_model_path))  # 加载最佳模型参数
        best_model = best_model.to(device)

        # 使用最佳模型在验证集上再次进行最终评估
        GP, GR, FS = test(best_model, val_iterator, device, logs)

        # 记录每折最终的评估分数
        all_gp_scores.append(GP)
        all_gr_scores.append(GR)
        all_fs_scores.append(FS)

    # 返回所有折的平均分数
    return np.mean(all_gp_scores), np.mean(all_gr_scores), np.mean(all_fs_scores)



def mian():

  args = parse_args()

  set_seed(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ids_log = log.create_log_id(args.save_log)
  log.logging_config(folder=args.save_log, name='log{:d}'.format(ids_log), no_console=False)
  logging.info(f'time: {time.asctime(time.localtime(time.time()))}')
  logging.info(args)

  GR,GP,FS = train_and_evaluate_with_kfold(args, device, logging)
  logging.info(f'Average GP: {GP:.5f} | Average GR: {GR:.5f} | Average FS: {FS:.5f}')


if __name__ == '__main__': # type: ignore
    
    mian()


# 读取reddit文件数据
def read_reddit_data(file_path = 'data/reddit_clean.pkl'):
    # 读取pkl文件
    reddit_data = pd.read_pickle(file_path)
    label  = [ user['label'] for user in reddit_data]
    users = [ user['user'] for user in reddit_data]
    return reddit_data,label,users

# 创建reddit数据集的词汇表集合
def get_vocabulary(reddit_data):
    # 创建一个空集合
    vocabulary = OrderedSet()
    words_id = {}
    id_words = {}
    # 遍历reddit数据集
    for posts in reddit_data:
        for post in posts['subreddit']:
            # 将文本分割为单词
            words = post.strip().split()
            # 将单词添加到词汇表集合中
            for word in words:
                vocabulary.add(word)

    # 统计词频并创建词汇表
    # word_counts = Counter(word_set)
    vocab = ["<PAD>", "<UNK>"] + list(vocabulary)
    words_id = {word: idx for idx, word in enumerate(vocab)}
    id_words = {idx: word for word, idx in words_id.items()}
    print(f"Total {len(vocab)} words found in all txt files")

    return vocab,words_id,id_words



# 数值化在batch传递的时候进行转化

# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx,reddit_data):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.reddit_data = reddit_data

    
    # 读取每个样本的文本
    def read_text(self, user,reddit_data):
        user_index = re.findall(r'\d+', user)
        user_index = int(user_index[0])
        user_posts = reddit_data[user_index]['subreddit']
        return user_posts



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text = self.read_text(text,self.reddit_data)

        # 将词转换为索引
        text_indices = []
        for post in text:
            if len(post) == 0:
                post_indices = []
            else:
                post_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in post]
            text_indices.append(post_indices)

        # print(text_indices)
        return text_indices, label


# 自定义 collate_fn 函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    # 找到批次中最大的帖子数量和最大的词数量
    max_num_posts = max(len(text) for text in texts)
    # max_num_words = max((len(post) for text in texts for post in text), default=1)
    max_num_words = min(512, max((len(post) for text in texts for post in text), default=1))


    batch_size = len(texts)
    # 初始化填充后的张量和掩码
    padded_texts = torch.zeros(batch_size, max_num_posts, max_num_words, dtype=torch.long)

    # 创建掩码
    post_masks = torch.zeros(batch_size, max_num_posts, max_num_words, dtype=torch.bool)
    text_masks = torch.zeros(batch_size, max_num_posts, dtype=torch.bool)

    for i, text in enumerate(texts):
        num_posts = len(text)
        text_masks[i, :num_posts] = 1  # 标记实际存在的帖子位置
        for j, post in enumerate(text):
            # num_words = len(post)
            num_words = min(len(post), max_num_words)  # 限制单词数量不超过 256
            if num_words > 0:
                # 实际每个帖子的词
                padded_texts[i, j, :num_words] = torch.tensor(post[:num_words])
                post_masks[i, j, :num_words] = 1  # 标记实际存在的词位置



    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels, text_masks, post_masks





    
# 设置随机种子
def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True






def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing GRU CLASS Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--save_log", type=str, default='./log_record/')
    parser.add_argument("--save_model_path", default='../check_point/', type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--filter_sizes", type=int, default=[2,3,4])
    parser.add_argument("--cnn_size", type=int, default=32)
    parser.add_argument("--gru_size", type=int, default=64)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--class_num", default=5, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(args)




# 训练模型
def train(model, iterator, optimizer,  device,logs,args):
    model.train()
    epoch_loss = 0
    out_result = []
    label_result = []
    for batch in tqdm(iterator, desc="Training"):
        text, label, text_masks, post_masks = batch
        text = text.to(device)
        label = label.to(device)
        text_masks = text_masks.to(device)
        post_masks = post_masks.to(device)
        optimizer.zero_grad()
        output = model(text, text_masks, post_masks)
        loss = loss_function(output, label, loss_type='ce', expt_type=args.class_num, scale=2)
        _, predicted = torch.max(output.data, 1)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        label_result.extend(label.cpu().detach().numpy().tolist())
        out_result.extend(predicted.cpu().detach().numpy().tolist())

    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f'Train Loss: {epoch_loss/len(iterator):.5f} | Train GP: {GP:.5f} | Train GR: {GR:.5f} | Train FS: {FS:.5f}')
    
    return epoch_loss / len(iterator),GP


# 评估模型
def evaluate(model, iterator,  device,logs,args):
    model.eval()
    epoch_loss = 0
    out_result = []
    label_result = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text, label, text_masks, post_masks = batch
            labels.append(label)
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)
            output = model(text, text_masks, post_masks)
            loss = loss_function(output, label, loss_type='ce', expt_type=args.class_num, scale=2)
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            label_result.extend(label.cpu().numpy().tolist())
            out_result.extend(predicted.cpu().numpy().tolist())
            
    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f'Val Loss: {epoch_loss/len(iterator):.5f} | Val GP: {GP:.5f} | Val GR: {GR:.5f} | Val FS: {FS:.5f}')
    return epoch_loss / len(iterator),GP

# 测试模型
def test(model, iterator, device,logs):
    model.eval()
    out_result = []
    label_result = []
    with torch.no_grad():
        for batch in iterator:
            text, label, text_masks, post_masks = batch
            text = text.to(device)
            label = label.to(device)
            text_masks = text_masks.to(device)
            post_masks = post_masks.to(device)
            output = model(text, text_masks, post_masks)
            _, predicted = torch.max(output.data, 1)
            label_result.extend(label.cpu().numpy().tolist())
            out_result.extend(predicted.cpu().numpy().tolist())

    GP,GR,FS,_ = gr_metrics(out_result, label_result)
    logs.info(f' test GP: {GP:.5f} | test GR: {GR:.5f} | test FS: {FS:.5f}')
    return GP


def mian():

  args = parse_args()

  set_seed(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ids_log = log.create_log_id(args.save_log)
  log.logging_config(folder=args.save_log, name='log{:d}'.format(ids_log), no_console=False)
  logging.info(f'time: {time.asctime(time.localtime(time.time()))}')
  logging.info(args)


  reddit_data,labels,users = read_reddit_data()
  vocabulary,words_id,id_words = get_vocabulary(reddit_data)
    # 划分训练集和测试集
  train_texts, temp_texts, train_labels, temp_labels = train_test_split(users, labels, stratify=labels,test_size=0.2, random_state = args.seed)
  # val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=args.seed)
  test_texts, val_texts, test_labels, val_labels = train_test_split(temp_texts, temp_labels, stratify=temp_labels, test_size=0.5, random_state=args.seed)

  # 创建数据集
  train_dataset = TextDataset(train_texts, train_labels, words_id,reddit_data)
  train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=4)

  val_dataset = TextDataset(val_texts, val_labels, words_id,reddit_data)
  val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=4)


  test_dataset = TextDataset(test_texts, test_labels, words_id,reddit_data)
  test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=4)


  model = models(args=args, vocab_size=len(vocabulary),device=device)
  model = model.to(device)

  optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


  for epoch in range(args.epochs):
      logging.info(f'Epoch: {epoch+1:02}')
      train_loss,_ = train(model, train_iterator, optimizer, device,logging,args)
      val_loss,_ = evaluate(model, val_iterator,  device,logging,args)
      GP = test(model, test_iterator, device,logging)



if __name__ == '__main__': # type: ignore
    
    mian()
