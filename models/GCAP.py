import torch
import torch.nn as nn
import torch.nn.functional as F


# 这个模型GRU+CNN+Attention/GRU+ATTENTION+PAIRMATCH

# 定义交互多层感知分类器模块
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, class_num,output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, class_num))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        result = self.mlp(x)
        return F.softmax(result, dim=1)




# 定义HAN模型分类器
# todo 添加将不同长度句子和不同单词数进行考虑
class GRU_CNN_Attention(nn.Module):
    '''层次注意力网络文档分类模型实现，词向量，句子向量'''
    def __init__(self,args, vocab_size,device,weights=None, is_pretrain=False):
        super(GRU_CNN_Attention, self).__init__()
        self.args = args
        self.device = device
        self.dropout = self.args.dropout
        self.gru_size = self.args.gru_size
        self.class_num = self.args.class_num
        self.embedding_dim = self.args.embedding_dim

        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, self.embedding_dim)        
        self.word_gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.gru_size,num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*self.gru_size, 1), requires_grad=True)   # 公式中的u(w)  
        self.word_fc = nn.Linear(2*self.gru_size, 2*self.gru_size)
        # 定义cnn
        self.convs = nn.ModuleList([nn.Conv2d(1,args.cnn_size, (k, self.gru_size*2)) for k in self.args.filter_sizes])
        self.conv_dims = len(self.args.filter_sizes) * args.cnn_size
        self.sentence_query = nn.Parameter(torch.Tensor(self.conv_dims, 1), requires_grad=True)   # 公式中的u(s)
        self.sentence_fc = nn.Linear(self.conv_dims, self.conv_dims)
        # todo 未对注意力进行初始化
        self._init_weights()

        # 文档分类,交互多层感知机其中的隐藏层维度为[conv_dims],可以考虑进行维度的变化。
        self.class_fc = MultiLayerPerceptron(self.conv_dims+self.gru_size*2, [self.gru_size], self.dropout,self.class_num)
        # self.class_fc = CrossNetwork(self.conv_dims, 2,self.dropout,self.class_num)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.word_query)  # 对 word_query 进行 Xavier 初始化
        nn.init.xavier_uniform_(self.sentence_query)  # 对 sentence_query 进行 Xavier 初始化

    
    
    # 处理成对相似性（优化）
    def compute_batch_commonality_optimized(self,batch_post_vectors, post_mask, max_recent_posts=20):
        batch_size = batch_post_vectors.size(0)
        embedding_dim = batch_post_vectors.size(2)
        batch_commonality = torch.zeros(batch_size, embedding_dim, device=batch_post_vectors.device)
        leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        for i in range(batch_size):
            post_vectors = batch_post_vectors[i]
            mask = post_mask[i].bool()
            real_post_vectors = post_vectors[mask]
            if real_post_vectors.size(0) <= max_recent_posts:
                recent_posts = real_post_vectors
            else:
                recent_posts = real_post_vectors[-max_recent_posts:]

            l = recent_posts.size(0)
            if l == 0:
                continue

            sum_vector = torch.sum(recent_posts, dim=0)
            sum_of_squares = torch.sum(recent_posts ** 2, dim=0)

            s_pair = (sum_vector ** 2 - sum_of_squares) / (2 * l)
            aggregated_features = leaky_relu(s_pair)

            batch_commonality[i] = aggregated_features

        return batch_commonality



    def forward(self, x,text_masks, post_masks ,use_gpu=False):  # x: b, sentence_num, sentence_len
            batch_size, sentence_num,sentence_len = x.size()
            x = x.view(-1, sentence_len)  # b*sentence_num, sentence_len
            post_masks = post_masks.view(-1, sentence_len)  # b*sentence_num, sentence_len
            # 计算每个帖子的实际词数
            word_lengths = post_masks.sum(dim=1)  # (batch_size * max_num_posts)
            non_zero_indices = (word_lengths > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) == 0:
                return torch.zeros(batch_size, self.fc.out_features)
        
            texts_non_zero = x[non_zero_indices]
            # print(texts_non_zero.shape)
            word_lengths_non_zero = word_lengths[non_zero_indices]
            
            # embed_x = self.word_embed(texts_non_zero)  # b*sentence_num , sentence_len, embedding_dim
            embed_x = self.word_embed(texts_non_zero)  # b*sentence_num , sentence_len, embedding_dim
            # embed_x = self.word_embed(x)  # b*sentence_num , sentence_len, embedding_dim
            embed_x_mask = post_masks[non_zero_indices]  # b*sentence_num, sentence_len

            # 使用 pack_padded_sequence 处理变长序列
            packed_words = nn.utils.rnn.pack_padded_sequence(
                embed_x, word_lengths_non_zero.cpu(), batch_first=True, enforce_sorted=False)
            # 通过词级别 RNN

            word_output, word_hidden = self.word_gru(packed_words)
            word_output, lengths = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)

            conv_output = [conv(word_output.unsqueeze(1)).squeeze(3) for conv in self.convs]

            # todo 进行attention 把具有指示性的核部分突出
            conv_Fmax = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_output]
            # -------------------------------------------------------------------------
            concat_conv = torch.cat(conv_Fmax, 1)

            # 开始直接进行attention,突出具有突出指示性性的帖子
            temp_post = torch.zeros(batch_size * sentence_num, self.conv_dims).to(self.device)
            temp_post[non_zero_indices] = concat_conv
            sentence_vector = temp_post.view(batch_size, sentence_num, self.conv_dims)



          # 计算u(it)
            word_attention = torch.tanh(self.word_fc(word_output))  # b*sentence_num, sentence_len, 2*gru_size
            weights = torch.matmul(word_attention, self.word_query)  # b*sentence_num, sentence_len, 1
            weights = F.softmax(weights, dim=1)   # b*sentence_num, sentence_len, 1
            x = x.unsqueeze(2)  # b*sentence_num, sentence_len, 1
            embed_x_mask = embed_x_mask.unsqueeze(2)  # b*sentence_num, sentence_len, 1
            # print(x.shape,'x')
            if use_gpu:
                # 去掉x中padding为0位置的attention比重
                weights = torch.where(embed_x_mask!=0, weights, torch.full_like(embed_x_mask, 0, dtype=torch.float).cuda()) #b*sentence_num, sentence_len, 1
            else:
                weights = torch.where(embed_x_mask!=0, weights, torch.full_like(embed_x_mask, 0, dtype=torch.float))

            # 将x中padding后的结果进行归一化处理，为了避免padding处的weights为0无法训练，加上一个极小值1e-4
            weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)  # b*sentence_num, sentence_len, 1   
            # todo 对每个帖子的词进行注意力机制，这部分的padd部分还未处理
            world_hidden = torch.sum(weights * word_output, dim=1)

            pair_post = torch.zeros(batch_size * sentence_num,world_hidden.size(1)).to(self.device)
            pair_post[non_zero_indices] = world_hidden
            pair_vector = pair_post.view(batch_size, sentence_num, world_hidden.size(1))

            # 计算公共性
            commonality = self.compute_batch_commonality_optimized(pair_vector, text_masks)
            post_lengths = text_masks.sum(dim=1)  # (batch_size)
            # 过滤掉长度为0的文本
            non_zero_post_indices = (post_lengths > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_post_indices) == 0:
                return torch.zeros(batch_size, self.class_fc.out_features)
            

            # 用注意力计算结果计算聚合文本向量
            texts_non_zero = sentence_vector[non_zero_post_indices]
            text_masks_non_zero = text_masks[non_zero_post_indices]
            # 计算ui
            sentence_attention = torch.tanh(self.sentence_fc(texts_non_zero))  # sentence_output: b, sentence_num, 2*gru_size
            # 计算句子注意力向量sentence_weights: a(i)
            sentence_weights = torch.matmul(sentence_attention, self.sentence_query)   # sentence_output: b, sentence_num, 1
            sentence_weights = F.softmax(sentence_weights, dim=1)   # b, sentence_num, 1

            text_masks_non_zero = text_masks_non_zero.unsqueeze(2)  # b, sentence_num, 1
            if use_gpu:
                sentence_weights = torch.where(text_masks_non_zero!=0, sentence_weights, torch.full_like(text_masks_non_zero, 0, dtype=torch.float).cuda())  
            else:
                sentence_weights = torch.where(text_masks_non_zero!=0, sentence_weights, torch.full_like(text_masks_non_zero, 0, dtype=torch.float))  # b, sentence_num, 1
 
            sentence_weights = sentence_weights / (torch.sum(sentence_weights, dim=1).unsqueeze(1) + 1e-4)  # b, sentence_num, 1 
            document_vector = torch.sum(sentence_weights * texts_non_zero, dim=1)   # b, sentence_num, 2*gru_size
            all_user_vectors = torch.concat([document_vector, commonality], dim=1)  # b, 2*gru_size + 2*gru_size

            document_class = self.class_fc(all_user_vectors)   # b, sentence_num, class_num
            return document_class   