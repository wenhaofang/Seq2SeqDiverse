import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import  pad_packed_sequence

import random

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.a = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim , 1 , bias = False)

    def forward(self, decoder_hiddens, encoder_outputs, encoder_masked):
        '''
        Params:
            decoder_hiddens: Torch Tensor (batch_size, hid_dim)
            encoder_outputs: Torch Tensor (batch_size, src_len, hid_dim)
            encoder_masked : Torch Tensor (batch_size, src_len)
        Return:
            scores         : Torch Tensor (batch_size, src_len)
            weight         : Torch Tensor (batch_size, hid_dim)
        '''
        decoder_hiddens = decoder_hiddens.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        concat = torch.cat((
            decoder_hiddens,
            encoder_outputs,
        ),  dim = 2)
        energy = torch.tanh   (self.a(concat))
        scores = torch.softmax(self.v(energy).squeeze(2).masked_fill(encoder_masked == 0, -1e10), dim = 1)
        weight = torch.bmm    (scores.unsqueeze(1), encoder_outputs).squeeze(1)
        return weight#, scores

class Encoder(nn.Module):
    def __init__(self, num_tok, pad_idx, emb_dim, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(num_tok, emb_dim, padding_idx = pad_idx)
        self.encoder   = nn.GRU      (emb_dim, hid_dim, batch_first = True)

    def forward(self, src, src_len):
        '''
        Params:
            src    : Torch Tensor (batch_size, seq_len)
            src_len: Torch Tensor (batch_size)
        Return:
            outputs: Torch Tensor (batch_size, seq_len , hid_dim)
            hiddens: Torch Tensor (batch_size, hid_dim)
        '''
        embedded = self.dropout(self.embedding(src))
        embedded = pack_padded_sequence(embedded, src_len.cpu(), batch_first = True, enforce_sorted = False)
        outputs, hiddens = self.encoder(embedded)
        outputs, lengths = pad_packed_sequence(outputs, batch_first = True)
        return outputs, hiddens.squeeze(0)

class Decoder(nn.Module):
    def __init__(self, num_tok, pad_idx, emb_dim, hid_dim, dropout):
        super(Decoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(num_tok , emb_dim , padding_idx = pad_idx)
        self.decoder   = nn.GRU      (emb_dim + hid_dim , hid_dim , batch_first = True)
        self.transform = nn.Linear   (hid_dim + emb_dim + hid_dim , num_tok)
        self.attention = Attention   (hid_dim , hid_dim)

    def forward(self, decoder_inputs, decoder_hiddens, encoder_outputs, encoder_masked):
        '''
        Params:
            decoder_inputs : Torch Tensor (batch_size)
            decoder_hiddens: Torch Tensor (batch_size, hid_dim)
            encoder_outputs: Torch Tensor (batch_size, src_len, hid_dim)
            encoder_masked : Torch Tensor (batch_size, src_len)
        Return:
            predictions    : Torch Tensor (batch_size, num_tok)
        '''
        embedded = self.dropout(self.embedding(decoder_inputs)).unsqueeze(1)
        weight = self.attention(decoder_hiddens, encoder_outputs, encoder_masked).unsqueeze(1)
        decoder_hiddens = decoder_hiddens . unsqueeze(0)
        decoder_outputs , decoder_hiddens = self.decoder(torch.cat((embedded, weight), dim = 2), decoder_hiddens)
        decoder_hiddens = decoder_hiddens . squeeze(0)
        predictions = self.transform(torch.cat((decoder_outputs.squeeze(1), embedded.squeeze(1), weight.squeeze(1)), dim = 1))
        return predictions, decoder_hiddens

class Seq2Seq(nn.Module):
    def __init__( self,
        enc_vocab_size, enc_pad_idx, enc_emb_dim, enc_hid_dim, enc_dropout,
        dec_vocab_size, dec_pad_idx, dec_emb_dim, dec_hid_dim, dec_dropout,
    ):
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = enc_vocab_size
        self.trg_vocab_size = dec_vocab_size
        self.src_pad_idx = enc_pad_idx
        self.trg_pad_idx = dec_pad_idx
        self.encoder = Encoder(enc_vocab_size, enc_pad_idx, enc_emb_dim, enc_hid_dim, enc_dropout)
        self.decoder = Decoder(dec_vocab_size, dec_pad_idx, dec_emb_dim, dec_hid_dim, dec_dropout)

    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        '''
        Params:
            src    : Torch Tensor (batch_size, src_len)
            src_len: Torch Tensor (batch_size)
            trg    : Torch Tensor (batch_size, trg_len)
        Return:
            outputs: Torch Tensor (batch_size, trg_len, trg_vocab_size)
        '''
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        encoder_outputs , encoder_hiddens = self.encoder(src, src_len)
        decoder_hiddens = encoder_hiddens
        encoder_masked  = src != self.src_pad_idx
        decoder_inputs  = trg[:, 0]
        decoder_outputs = torch.zeros((batch_size, trg_len, self.trg_vocab_size), device = src.device)
        for t in range(trg_len):
            decoder_output , decoder_hiddens = self.decoder(decoder_inputs, decoder_hiddens, encoder_outputs, encoder_masked)
            decoder_inputs = trg[:, t] if random.random() < teacher_forcing_ratio else decoder_output.argmax(1)
            decoder_outputs[:, t] = decoder_output
        return decoder_outputs

    def predict(self, src, src_len, max_len, sos_idx, eos_idx, decoding_algorithm, T = None, K = None, P = None):
        '''
        Params:
            src    : Torch Tensor (batch_size, src_len)
            src_len: Torch Tensor (batch_size)
            max_len: Int
            sos_idx: Int
            eos_idx: Int
            decoding_algorithm: String
        '''
        assert src.shape[0] == 1
        assert decoding_algorithm in ['temperature_sampling', 'top_k_sampling', 'top_p_sampling']

        if  decoding_algorithm == 'temperature_sampling':
            assert T != None
        if  decoding_algorithm == 'top_k_sampling':
            assert K != None
        if  decoding_algorithm == 'top_p_sampling':
            assert P != None

        encoder_outputs , encoder_hiddens = self.encoder(src, src_len)
        decoder_hiddens = encoder_hiddens
        encoder_masked  = src != self.src_pad_idx
        decoder_inputs  = torch.full((src.size(0),), sos_idx, dtype = src.dtype, device = src.device)

        outputs = []

        if  (
            decoding_algorithm == 'temperature_sampling' or
            decoding_algorithm == 'top_k_sampling' or
            decoding_algorithm == 'top_p_sampling' # nucleus_sampling
        ):
            for _ in range(max_len):
                decoder_output , decoder_hiddens = self.decoder(decoder_inputs, decoder_hiddens, encoder_outputs, encoder_masked)
                decoder_inputs = self.t_dis(decoder_output, T) if decoding_algorithm == 'temperature_sampling' else \
                                 self.k_dis(decoder_output, K) if decoding_algorithm == 'top_k_sampling' else \
                                 self.p_dis(decoder_output, P) if decoding_algorithm == 'top_p_sampling' else \
                                 None

                if  eos_idx != decoder_inputs.item():
                    outputs.append(decoder_inputs.item())
                else:
                    break

        return outputs

    def t_dis(self, logits, t):
        '''
        Params:
            logits: Torch Tensor (batch_size, vocab_size)
            t:
                t 越小，分布越突出，特殊情况 t → 0, greedy sampling
                t 越大，分布越平缓，特殊情况 t → ∞, random sampling
        Return:
            output: Torch Tensor (batch_size)
        '''
        assert t > 0

        return torch.multinomial(torch.softmax(logits / t, dim = -1), num_samples = 1).view(-1)

    def k_dis(self, logits, k):
        '''
        Params:
            logits: Torch Tensor (batch_size, vocab_size), k: 概率降序排列，按序取值使其累加数量大于等于 k
        Return:
            output: Torch Tensor (batch_size)
        '''
        assert k > 0

        sorted_value, sorted_index = torch.sort(logits, dim = -1, descending = True)

        arange_tensor = torch.arange(logits.shape[1]      , device = logits.device).unsqueeze(0).repeat(logits.shape[0], 1)
        kvalue_tensor = torch.tensor(logits.shape[1] * [k], device = logits.device).unsqueeze(0).repeat(logits.shape[0], 1)
        masked_tensor = arange_tensor >= kvalue_tensor

        sorted_logits = torch.masked_scatter(sorted_value, masked_tensor, torch.full((sorted_value.shape), -1e13, device = logits.device))
        sorted_output = torch.multinomial(torch.softmax(sorted_logits, dim = -1), num_samples = 1)
        output = torch.gather(sorted_index, dim = 1, index = sorted_output).view(-1)

        return output

    def p_dis(self, logits, p):
        '''
        Params:
            logits: Torch Tensor (batch_size, vocab_size), p: 概率降序排列，按序取值使其累加概率大于等于 p
        Return:
            output: Torch Tensor (batch_size)
        '''
        assert 0 < p <= 1

        sorted_value, sorted_index = torch.sort(logits, dim = -1, descending = True)

        sofmax_tensor = torch.softmax(sorted_value , dim = -1)
        cumsum_tensor = torch.cumsum (sofmax_tensor, dim = -1)
        masked_tensor = cumsum_tensor > p

        sorted_logits = torch.masked_scatter(sorted_value, masked_tensor, torch.full((sorted_value.shape), -1e13, device = logits.device))
        sorted_output = torch.multinomial(torch.softmax(sorted_logits, dim = -1), num_samples = 1)
        output = torch.gather(sorted_index, dim = 1, index = sorted_output).view(-1)

        return output

def get_module(
    option,
    src_vocab_size, src_pad_idx,
    trg_vocab_size, trg_pad_idx,
):
    return Seq2Seq(
        src_vocab_size, src_pad_idx, option.emb_dim, option.hid_dim, option.dropout,
        trg_vocab_size, trg_pad_idx, option.emb_dim, option.hid_dim, option.dropout,
    )

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab_size = 10000
    trg_vocab_size = 12000

    src_pad_idx = 0
    trg_pad_idx = 0

    trg_sos_idx = 3
    trg_eos_idx = 2

    src_seq_len = 18
    trg_seq_len = 20

    module = get_module(option, src_vocab_size, src_pad_idx, trg_vocab_size, trg_pad_idx)

    src = torch.randint(0, src_vocab_size, (option.batch_size, src_seq_len)).long()
    trg = torch.randint(0, trg_vocab_size, (option.batch_size, trg_seq_len)).long()
    src_len = torch.tensor([src_seq_len] *  option.batch_size).long()

    outputs = module(src, src_len, trg, 0.5)

    print(outputs.shape) # (batch_size, trg_seq_len, trg_vocab_size)

    src = torch.randint(0, src_vocab_size, (1, src_seq_len)).long()
    trg = torch.randint(0, trg_vocab_size, (1, trg_seq_len)).long()
    src_len = torch.tensor([src_seq_len] *  1).long()

    t_outputs = module.predict(src, src_len, option.max_seq_len, trg_sos_idx, trg_eos_idx, 'temperature_sampling', T = option.T)
    k_outputs = module.predict(src, src_len, option.max_seq_len, trg_sos_idx, trg_eos_idx, 'top_k_sampling', K = option.K)
    p_outputs = module.predict(src, src_len, option.max_seq_len, trg_sos_idx, trg_eos_idx, 'top_p_sampling', P = option.P)

    print(type(t_outputs), len(t_outputs)) # List
    print(type(k_outputs), len(k_outputs)) # List
    print(type(p_outputs), len(p_outputs)) # List
