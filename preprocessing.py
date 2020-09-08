import jieba
import re

class preprocessing():
    __PAD__ = 0   # 填充词汇
    __GO__ = 1    # 句子开始
    __EOS__ = 2   # 句子结束
    __UNK__ = 3   # 未知词(低频的词替换为UNK)
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']
    def __init__(self):
        #self.encoderFile = "/home/yanwii/Python/NLP/seq2seq/seq2seq_no_buckets/preprocessing/MySeq2seq/Data/alldata_ask.txt"
        #self.decoderFile = '/home/yanwii/Python/NLP/seq2seq/seq2seq_no_buckets/preprocessing/MySeq2seq/Data/alldata_answer.txt'
        #self.savePath = '/home/yanwii/Python/NLP/seq2seq/seq2seq_pytorch/data/'
        self.encoderFile = "./data/question.txt"
        self.decoderFile = "./data/answer.txt"
        self.savePath = './data/'
        
        jieba.load_userdict("./data/supplementvocab.txt")
    
    def wordToVocabulary(self, originFile, vocabFile, segementFile):
        vocabulary = []
        sege = open(segementFile, "w", encoding='utf8')
        with open(originFile, 'r', encoding='utf8') as en:
            for sent in en.readlines():
                # 去标点
                if "enc" in segementFile:
                    #sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
                    sentence = sent.strip()
                    words = jieba.lcut(sentence)
                    # print(words)
                else:
                    words = jieba.lcut(sent.strip())
                vocabulary.extend(words)
                for word in words:
                    sege.write(word+" ")
                sege.write("\n")
        sege.close()

        # 去重并存入词典
        vocabulary_map = {}
        for ind, val in enumerate(vocabulary):
            vocabulary_map[val] = ind
        vocab_file = open(vocabFile, "w", encoding='utf8')
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=lambda x: vocabulary_map[x])
        _vocabulary = self.vocab + _vocabulary
        for index, word in enumerate(_vocabulary):
            vocab_file.write(word+"\n")
        vocab_file.close()

    def toVec(self, segementFile, vocabFile, doneFile):
        word_dicts = {}
        vec = []
        with open(vocabFile, "r", encoding='utf8') as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(doneFile, "w", encoding='utf8')
        if "enc.vec" in doneFile:
            f.write("3 3 3 3\n")
            f.write("3\n")
        elif "dec.vec" in doneFile:
            f.write(str(word_dicts.get("other", 3))+"\n")
            f.write(str(word_dicts.get("other", 3))+"\n")
        with open(segementFile, "r", encoding='utf8') as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word))+" ")
                f.write("\n")
        f.close()
            

    def main(self):
        # 获得字典
        self.wordToVocabulary(self.encoderFile, self.savePath+'enc.vocab', self.savePath+'enc.segement')
        self.wordToVocabulary(self.decoderFile, self.savePath+'dec.vocab', self.savePath+'dec.segement')
        # 转向量
        self.toVec(self.savePath+"enc.segement", 
                   self.savePath+"enc.vocab", 
                   self.savePath+"enc.vec")
        self.toVec(self.savePath+"dec.segement", 
                   self.savePath+"dec.vocab", 
                   self.savePath+"dec.vec")


if __name__ == '__main__':
    pre = preprocessing()
    pre.main()
