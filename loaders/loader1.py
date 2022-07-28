import os
import copy
import spacy
import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

class QuoraDataset(Dataset):
    def __init__(
        self,
        is_train,
        min_freq,
        max_numb,
        data_path,
        src_vocab = None,
        trg_vocab = None,
    ):
        super(QuoraDataset, self).__init__()
        if  not is_train:
            assert src_vocab != None
            assert trg_vocab != None
        self._spacy_en = spacy.load('en_core_web_sm')
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        all_datas = self.read_data(data_path)
        src_sents = list(map(lambda x: x[0], all_datas))
        trg_sents = list(map(lambda x: x[1], all_datas))
        src_words = list(map(lambda s: self.tokenize_en(s), src_sents))
        trg_words = list(map(lambda s: self.tokenize_en(s), trg_sents))
        self.src_vocab = self.build_vocab(src_words, min_freq, max_numb) if is_train else src_vocab
        self.trg_vocab = self.build_vocab(trg_words, min_freq, max_numb) if is_train else trg_vocab
        src_w_ids = list(map(lambda w: self.encode(w, self.src_vocab), src_words))
        trg_w_ids = list(map(lambda w: self.encode(w, self.trg_vocab), trg_words))
        self.all_datas = self.aggregate(src_w_ids, trg_w_ids)

    def read_data(self, data_path):
        d = pd.read_csv(data_path, sep = '\t')
        d = list(zip(
            d['q1'].tolist(),
            d['q2'].tolist(),
        ))
        return d

    def tokenize_en(self, text):

        return [token.text.lower() for token in self._spacy_en.tokenizer(text)]

    def build_vocab(self, data, min_freq, max_numb):
        counter = {}
        for sent in data:
            for word in sent:
                counter[word] = counter.get(word, 0) + 1

        word_list = filter(lambda x : x[1] >= min_freq, counter.items())
        word_list = sorted(word_list, key = lambda x : x[1], reverse = True)[:max_numb - 4]

        words = [word for word, count in word_list]
        words.insert(0, self.SOS_TOKEN)
        words.insert(0, self.EOS_TOKEN)
        words.insert(0, self.UNK_TOKEN)
        words.insert(0, self.PAD_TOKEN)

        vocab = {}
        vocab['id2word'] = {idx: word for idx, word in enumerate(words)}
        vocab['word2id'] = {word: idx for idx, word in enumerate(words)}
        vocab['special'] = {
            'SOS_TOKEN': self.SOS_TOKEN,
            'EOS_TOKEN': self.EOS_TOKEN,
            'UNK_TOKEN': self.UNK_TOKEN,
            'PAD_TOKEN': self.PAD_TOKEN
        }

        return vocab

    def encode(self, tokens, vocab):
        SOS_ID = vocab['word2id'].get(self.SOS_TOKEN)
        EOS_ID = vocab['word2id'].get(self.EOS_TOKEN)
        UNK_ID = vocab['word2id'].get(self.UNK_TOKEN)

        return [SOS_ID] + [vocab['word2id'].get(token, UNK_ID) for token in tokens] + [EOS_ID]

    def aggregate(self, src_w_ids, trg_w_ids):
        return [(s_ids, t_ids) for s_ids, t_ids in zip(src_w_ids, trg_w_ids)]

    def get_src_voacb(self):
        return copy.deepcopy(self.src_vocab)

    def get_trg_vocab(self):
        return copy.deepcopy(self.trg_vocab)

    def __len__(self):
        return len(self.all_datas)

    def __getitem__(self, idx):
        return self.all_datas[idx]

def collate_fn(batch_data, src_pad_id, trg_pad_id):
    src = [data[0] for data in batch_data]
    trg = [data[1] for data in batch_data]

    datas = [(s, t, len(s)) for s, t in zip(src, trg)]
    datas = sorted(datas, key = lambda x: x[2] , reverse = True)

    src = [data[0] for data in datas]
    trg = [data[1] for data in datas]

    src_len = [data[2] for data in datas]

    src = [torch.tensor(seq, dtype = torch.long) for seq in src]
    trg = [torch.tensor(seq, dtype = torch.long) for seq in trg]

    src = pad_sequence(src, batch_first = True, padding_value = src_pad_id)
    trg = pad_sequence(trg, batch_first = True, padding_value = trg_pad_id)

    src_len = torch.tensor(src_len, dtype = torch.long)

    return (src, trg, src_len)

def get_loader(option):
    train_file = os.path.join(option.targets_path, option.train_file)
    valid_file = os.path.join(option.targets_path, option.valid_file)
    test_file  = os.path.join(option.targets_path, option.test_file )

    train_dataset = QuoraDataset(True , option.min_freq, option.max_numb, train_file)
    valid_dataset = QuoraDataset(False, option.min_freq, option.max_numb, valid_file, train_dataset.get_src_voacb(), train_dataset.get_trg_vocab())
    test_dataset  = QuoraDataset(False, option.min_freq, option.max_numb, test_file , train_dataset.get_src_voacb(), train_dataset.get_trg_vocab())

    SRC_PAD_ID = train_dataset.src_vocab['word2id'].get(train_dataset.src_vocab['special']['PAD_TOKEN'])
    TRG_PAD_ID = train_dataset.trg_vocab['word2id'].get(train_dataset.trg_vocab['special']['PAD_TOKEN'])

    train_loader = DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True , collate_fn = lambda x: collate_fn(x, SRC_PAD_ID, TRG_PAD_ID))
    valid_loader = DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = True , collate_fn = lambda x: collate_fn(x, SRC_PAD_ID, TRG_PAD_ID))
    test_loader  = DataLoader(test_dataset , batch_size = option.batch_size, shuffle = False, collate_fn = lambda x: collate_fn(x, SRC_PAD_ID, TRG_PAD_ID))

    return (train_dataset.get_src_voacb(), train_dataset.get_trg_vocab(), train_loader, valid_loader, test_loader)

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab, trg_vocab, train_loader, valid_loader, test_loader = get_loader(option)

    # vocab
    print(type(src_vocab), src_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(type(trg_vocab), trg_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(len(src_vocab['word2id']), len(src_vocab['id2word'])) # 10701 10701
    print(len(trg_vocab['word2id']), len(trg_vocab['id2word'])) # 10702 10702

    # dataloader
    print(len(train_loader.dataset)) # 134336
    print(len(valid_loader.dataset)) # 7463
    print(len(test_loader .dataset)) # 7464
    for mini_batch in train_loader:
        src, trg, src_len = mini_batch
        print(src.shape)     # (batch_size, src_len)
        print(trg.shape)     # (batch_size, trg_len)
        print(src_len.shape) # (batch_size)
        break
