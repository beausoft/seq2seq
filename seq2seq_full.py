# -*- coding: UTF-8 -*-
import math
import os
import random
import sys
import time

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data

USE_CUDA = torch.cuda.is_available()
SOS_token = 2
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs)   # 得到的词向量，每一行代表一个词
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len, batch_size, _ = encoder_outputs.size()

        attn_energies = torch.zeros((seq_len, batch_size))  # B x 1 x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # 按照批次依次计算得分
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim=0).transpose(1, 0).unsqueeze(1)

    def score(self, hidden, encoder_output):
        energys = list()
        if self.method == 'dot':
            for i in range(len(encoder_output)):
                energys.append(torch.dot(hidden[i].view(-1), encoder_output[i].view(-1)))
            return torch.stack(energys)
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            for i in range(len(encoder_output)):
                energys.append(torch.dot(hidden[i].view(-1), encoder_output[i].view(-1)))
            return torch.stack(energys)


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, self.max_length)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):

        word_embedded = self.embedding(word_input)  # .view(1, 1, -1) # S=1 x B x N

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        # bmm https://blog.csdn.net/qq_40178291/article/details/100302375
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)
        #output = self.out(torch.cat((rnn_output, context), 1))
        return output, context, hidden, attn_weights

class CorpusDataset(data.Dataset):

    def __init__(self, encode_file = './data/enc.vec', decode_file = './data/dec.vec', encoding = 'UTF-8-sig'):
        self.enc_vec = []
        self.dec_vec = []
        with open(encode_file, 'r', encoding=encoding) as enc:
            line = enc.readline()
            while line:
                self.enc_vec.append(line.strip().split())
                line = enc.readline()
        with open(decode_file, 'r', encoding=encoding) as dec:
            line = dec.readline()
            while line:
                self.dec_vec.append(line.strip().split())
                line = dec.readline()

    def __getitem__(self, index):
        input = self.enc_vec[index]
        target = self.dec_vec[index]
        input = [int(i) for i in input]
        target = [int(i) for i in target]
        target.insert(0, SOS_token)    # 句子起始标识
        target.append(EOS_token)       # 句子结束标识
        input = torch.tensor(input, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return input, target
    
    def __len__(self):
        length = max(len(self.enc_vec), len(self.dec_vec))
        return length

class RnnDataloader(data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, drop_last=False):
        if collate_fn is None:
            collate_fn = self._pad_seq_collate_fn
        super(RnnDataloader, self).__init__(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last)
    
    def _pad_seq_collate_fn(self, datas):
        inputs = []
        targets = []
        for x, y in datas:
            inputs.append(x)
            targets.append(y)
        inputs = rnn_utils.pad_sequence(inputs)
        targets = rnn_utils.pad_sequence(targets)
        return inputs, targets

class SEQ2SEQ(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, n_layers=2, dropout_p=0.05, max_length=64):
        super(SEQ2SEQ, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 初始化encoder和decoder
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = AttnDecoderRNN('general', self.hidden_size, self.output_size, self.n_layers, self.dropout_p, self.max_length)

    def _forward_encoding(self, inputs):
        _, batch_size = inputs.size()
        # 解码器
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)
        return encoder_outputs, encoder_hidden
    
    def _forward_decoding(self, decoder_input, decoder_context, decoder_hidden, encoder_outputs):
        _, batch_size, _ = encoder_outputs.size()
        if decoder_context is None:
            decoder_context = torch.zeros(batch_size, self.decoder.hidden_size)           # 创建编码器上下文
            if USE_CUDA:
                decoder_context = decoder_context.cuda()
        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        return decoder_output, decoder_context, decoder_hidden, decoder_attention
    
    def forward(self, cmd, *inputs):
        if cmd == 'encoding':
            return self._forward_encoding(*inputs)
        elif cmd == 'decoding':
            return self._forward_decoding(*inputs)
        else:
            raise NotImplementedError
            


class Master(object):

    def __init__(self):
        self.input_size = 1121
        self.output_size = 1830
        self.hidden_size = 128
        self.layers = 2
        self.dropout = 0.05
        self.max_length = 256
        self.checkpoint = './model/seq2seq_full.pkl'

        self.model = SEQ2SEQ(self.input_size, self.output_size, self.hidden_size, self.layers, self.dropout, self.max_length)
        if USE_CUDA:
            self.model.cuda()
            # self.model = nn.DataParallel(self.model)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint)

    def load_model(self):
        if os.path.exists(self.checkpoint) and os.path.isfile(self.checkpoint):
            self.model.load_state_dict(torch.load(self.checkpoint))
        else:
            print('No checkpoint')
    
    def _get_model(self):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        return model


class Trainer(Master):

    def __init__(self):
        super(Trainer, self).__init__()
        self.epoches = 30
        self.batch_size = 16
        self.show_epoch = 10
        self.clip = 5.0   # 梯度裁剪参数（防止梯度爆炸）
        self.teacher_forcing_ratio = 0.5    # 使用teacher_forcing技术的几率
        
        self.dataset = CorpusDataset()
        self.dataloader = RnnDataloader(self.dataset, self.batch_size, shuffle=True)

        self.encoder_optimizer = optim.Adam(self._get_model().encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self._get_model().decoder.parameters(), lr=0.001)
        self.criterion = nn.NLLLoss()

    def train(self):
        self.load_model()
        self.model.train()
        total_loss = 0
        total_count = 0
        batch_length = len(self.dataloader)
        for epoch in range(self.epoches):
            epoch_loss = 0
            epoch_count = 0
            every_time_count = 0
            for loss_val, outputs, targets, every_time in self._epoch_train():
                total_loss += loss_val
                epoch_loss += loss_val
                total_count += 1
                epoch_count += 1
                every_time_count += every_time

                if epoch_count % self.show_epoch == 0 or epoch_count == batch_length:
                    print('epoch: {} - [{}/{}], time: {:.3f}, total loss: {:.6f}, loss: {:.6f}'
                        .format(epoch + 1, 
                            epoch_count,
                            batch_length,
                            every_time_count / epoch_count,
                            total_loss / total_count, 
                            epoch_loss / epoch_count))
                    self.save_model()

    def _epoch_train(self):
        for i, (inputs, targets) in enumerate(self.dataloader):
            start = time.time()
            if USE_CUDA:
                inputs = inputs.cuda()
                targets = targets.cuda()
            # 优化器梯度置零
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            # 对输入词索引进行编码
            encoder_outputs, encoder_hidden = self.model('encoding', inputs)

            target_length = targets.shape[0]

            decoder_input = None
            decoder_context = None
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

            decoder_outputs = []

            loss = None
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            if use_teacher_forcing:
                for di in range(target_length):
                    decoder_input = targets[di].unsqueeze(0)    # 使用正确的target作为下一次的输入，我们称之为teacher forcing技术
                    decoder_output, decoder_context, decoder_hidden, _ = self.model('decoding', decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                    if loss is None:
                        loss = self.criterion(decoder_output, targets[di])
                    else:
                        loss += self.criterion(decoder_output, targets[di])
                    decoder_outputs.append(decoder_output.unsqueeze(0))
            else:
                for di in range(target_length):
                    if di == 0:
                        decoder_input = targets[di].unsqueeze(0)   # 因为第一个就是句子起始标识
                    decoder_output, decoder_context, decoder_hidden, _ = self.model('decoding', decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                    if loss is None:
                        loss = self.criterion(decoder_output, targets[di])
                    else:
                        loss += self.criterion(decoder_output, targets[di])
                    decoder_outputs.append(decoder_output.unsqueeze(0))
                    topv, topi = decoder_output.data.topk(1)
                    decoder_input = topi.view(decoder_input.shape)   # 直接使用网络输出的target作为下一次输入

            loss.backward()   # 向后传播梯度
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self._get_model().encoder.parameters(), self.clip)
                torch.nn.utils.clip_grad_norm_(self._get_model().decoder.parameters(), self.clip)
            # 更新参数
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            decoder_outputs = torch.cat(decoder_outputs, 0)
            loss_val = loss.item() / target_length
            stop = time.time()
            yield loss_val, decoder_outputs, targets, stop - start


trainer = Trainer()
trainer.train()
