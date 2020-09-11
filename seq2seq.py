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

        # .view(1, 1, -1) # S=1 x B x N
        word_embedded = self.embedding(word_input)

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


class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()
        self.max_epoches = 454609 * 10
        self.batch_index = 0
        self.GO_token = 2
        self.EOS_token = 1
        self.input_size = 1121
        self.output_size = 1830
        self.hidden_size = 128
        self.max_length = 256
        self.show_epoch = 1
        self.use_cuda = USE_CUDA
        self.model_path = "./model/"
        self.n_layers = 2
        self.dropout_p = 0.05
        self.beam_search = False
        self.top_k = 5
        self.alpha = 0.5
        self.batch_size = 16

        self.enc_vec = []
        self.dec_vec = []

        # 初始化encoder和decoder
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = AttnDecoderRNN('general', self.hidden_size, self.output_size, self.n_layers, self.dropout_p, self.max_length)

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.NLLLoss()

    def loadData(self):
        with open("./data/enc.vec", encoding='UTF-8-sig') as enc:
            line = enc.readline()
            while line:
                self.enc_vec.append(line.strip().split())
                line = enc.readline()

        with open("./data/dec.vec", encoding='UTF-8-sig') as dec:
            line = dec.readline()
            while line:
                self.dec_vec.append(line.strip().split())
                line = dec.readline()

    def next(self, batch_size, eos_token=1, go_token=2, shuffle=False):
        inputs = []
        targets = []

        if shuffle:
            ind = random.choice(range(len(self.enc_vec)))
            enc = [self.enc_vec[ind]]
            dec = [self.dec_vec[ind]]
        else:
            if self.batch_index+batch_size >= len(self.enc_vec):
                enc = self.enc_vec[self.batch_index:]
                dec = self.dec_vec[self.batch_index:]
                self.batch_index = 0
            else:
                enc = self.enc_vec[self.batch_index:self.batch_index+batch_size]
                dec = self.dec_vec[self.batch_index:self.batch_index+batch_size]
                self.batch_index += batch_size
        for index in range(len(enc)):
            enc_sub = enc[index][:self.max_length] if len(enc[index]) > self.max_length else enc[index]
            dec_sub = dec[index][:self.max_length] if len(dec[index]) > self.max_length else dec[index]

            enc_sub = [int(i) for i in enc_sub]
            dec_sub = [int(i) for i in dec_sub]
            dec_sub.append(eos_token)

            inputs.append(torch.tensor(enc_sub, dtype=torch.long))
            targets.append(torch.tensor(dec_sub, dtype=torch.long))

        inputs = rnn_utils.pad_sequence(inputs)
        targets = rnn_utils.pad_sequence(targets)

        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return inputs, targets

    def train(self):
        self.loadData()
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.next(self.batch_size, shuffle=False)
            loss, logits = self.step(inputs, targets, self.max_length)
            loss_track.append(loss)
            _, v = torch.topk(logits, 1)
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-"*50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop-start))
                print("    avg_loss:", np.array(
                    loss_track).sum() / len(loss_track))
                loss_track = []
                torch.save(self.state_dict(), self.model_path+'params.pkl')
        torch.save(self.state_dict(), self.model_path+'params.pkl')

    def step(self, input_variable, target_variable, max_length):
        teacher_forcing_ratio = 0.1
        clip = 5.0
        loss = 0  # Added onto for each word

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length, batch_size = input_variable.size()
        target_length = target_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        # torch.LongTensor([[SOS_token]])
        decoder_input = torch.zeros((1, batch_size), dtype=torch.long) + SOS_token
        decoder_context = torch.zeros(batch_size, self.decoder.hidden_size)
        # Use last hidden state from encoder to start decoder
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di].unsqueeze(0)    # 使用正确的target作为下一次的输入，我们称之为teacher forcing技术
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                # ni = topi[0][0]

                # decoder_input = torch.LongTensor([[ni]])
                # if USE_CUDA:
                #     decoder_input = decoder_input.cuda()
                decoder_input = topi.view(decoder_input.shape)   # 直接使用网络输出的target作为下一次输入
                # if ni == EOS_token:
                #     break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)
        return loss.item() / target_length, decoder_outputs

    def make_infer_fd(self, input_vec):
        inputs = []
        enc = input_vec[:self.max_length] if len(input_vec) > self.max_length else input_vec
        inputs.append(enc)
        inputs = torch.LongTensor(inputs).transpose(1, 0).contiguous()
        if USE_CUDA:
            inputs = inputs.cuda()
        return inputs

    def predict(self):
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        # 加载字典
        str_to_vec = {}
        with open("./data/enc.vocab", encoding='UTF-8-sig') as enc_vocab:
            for index, word in enumerate(enc_vocab.readlines()):
                str_to_vec[word.strip()] = index

        vec_to_str = {}
        with open("./data/dec.vocab", encoding='UTF-8-sig') as dec_vocab:
            for index, word in enumerate(dec_vocab.readlines()):
                vec_to_str[index] = word.strip()

        self.encoder.eval()
        self.decoder.eval()

        while True:
            input_strs = input("me > ")
            # 字符串转向量
            segement = jieba.lcut(input_strs)
            input_vec = [str_to_vec.get(i, 3) for i in segement]
            input_vec = self.make_infer_fd(input_vec)

            # inference
            if self.beam_search:
                samples = self.beamSearchDecoder(input_vec)
                for sample in samples:
                    outstrs = []
                    for i in sample[0]:
                        if i == 1:
                            break
                        outstrs.append(vec_to_str.get(i, "Un"))
                    print("ai > ", "".join(outstrs), sample[3])
            else:
                logits = self.infer(input_vec)
                _, v = torch.topk(logits, 1)
                pre = v.cpu().data.numpy().T.tolist()[0][0]
                outstrs = []
                for i in pre:
                    if i == 1:
                        break
                    outstrs.append(vec_to_str.get(i, "Un"))
                print("ai > ", "".join(outstrs))

    def infer(self, input_variable):
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = torch.LongTensor([[SOS_token]])
        decoder_context = torch.zeros(1, self.decoder.hidden_size)
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output.unsqueeze(0))
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = torch.LongTensor([[ni]])  # Chosen word is next input
            if USE_CUDA:
                decoder_input = decoder_input.cuda()
            if ni == EOS_token:
                break

        decoder_outputs = torch.cat(decoder_outputs, 0)
        return decoder_outputs

    def tensorToList(self, tensor):
        return tensor.cpu().data.numpy().tolist()[0]

    def beamSearchDecoder(self, input_variable):
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = torch.LongTensor([[SOS_token]])
        decoder_context = torch.zeros(1, self.decoder.hidden_size)
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        topk = decoder_output.data.topk(self.top_k)
        samples = [[] for i in range(self.top_k)]
        dead_k = 0
        final_samples = []
        for index in range(self.top_k):
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context, decoder_hidden, decoder_attention, encoder_outputs]

        for _ in range(self.max_length):
            tmp = []
            for index in range(len(samples)):
                tmp.extend(self.beamSearchInfer(samples[index], index))
            samples = []

            # 筛选出topk
            df = pd.DataFrame(tmp)
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores",
                          "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
            sequence_len = df.sequence.apply(lambda x: len(x))
            df['ave_scores'] = df['fin_scores'] / sequence_len
            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)
            df = df[:(self.top_k-dead_k)]
            for index in range(len(df)):
                # https://blog.csdn.net/tianjueshou/article/details/80850455
                # group = df.ix[index]
                # group = df.iloc[index]
                group = df.loc[index]
                if group.tolist()[0][-1] == 1:
                    final_samples.append(group.tolist())
                    df = df.drop([index], axis=0)
                    dead_k += 1
                    print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:
                break

        if len(final_samples) < self.top_k:
            final_samples.extend(samples[:(self.top_k-dead_k)])
        return final_samples

    def beamSearchInfer(self, sample, k):
        samples = []
        decoder_input = torch.LongTensor([[sample[0][-1]]])
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        sequence, pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

        # choose topk
        topk = decoder_output.data.topk(self.top_k)
        for k in range(self.top_k):
            topk_prob = topk[0][0][k]
            topk_index = int(topk[1][0][k])
            pre_scores += topk_prob
            fin_scores = pre_scores - (k - 1) * self.alpha
            samples.append([sequence+[topk_index], pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
        return samples

    def retrain(self):
        try:
            os.remove(self.model_path)
        except Exception as e:
            pass
        self.train()


if __name__ == '__main__':
    seq = seq2seq()
    if sys.argv[1] == 'train':
        seq.train()
    elif sys.argv[1] == 'predict':
        seq.predict()
    elif sys.argv[1] == 'retrain':
        seq.retrain()
