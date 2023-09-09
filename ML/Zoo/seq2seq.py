##### seq2seq + attention #####
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, dec_hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = True):
        """
        initializing a GRU encoder
        :param input_size: x, should be (B,H,F)
        :param embedding_size: the size to embed to
        :param hidden_size: the size of the hidden vectors
        :param dec_hidden_size: the size of the decoder's hidden vector, to be used as the output of the FC layer
        :param num_layers: numbers of stacked GRU blocks
        :param bidirectional: boolean indicating if the GRU is bidirectional
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.embedding = nn.Linear(self.input_size, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, self.dec_hidden_size)

    def forward(self, enc_input: torch.Tensor):
        """
        takes in an input tensor and returns
        	- all outputs [batch_size, seq_size, hidden_size * num_directions].
        	 each item in outputs is a concatenated forward and backward hidden state:
        	   [:, [[h_1->,h_L<-],[h_2->,h_(L-1)<-],...[h_L->,h_1<-]]
            -  last hidden [num_layers * num_directions,batch_size, hidden_size]
        """

        embedded = self.embedding(enc_input)
        outputs, last_hidden = self.rnn(embedded)
        fwd_last_hidden, bwd_last_hidden = last_hidden[-2, :, :], last_hidden[-1, :, :]
        hidden_cat = torch.cat((fwd_last_hidden, bwd_last_hidden), dim=1)
        # now our hidden has shape [batch_size, hidden_size * num_directions] and after FC: [batch_size, dec_hid_size]
        # we cast to dec_hidden_size as this hidden state is the initial hidden state of the decoder!
        return outputs, torch.tanh(self.fc(hidden_cat))


class Attention(nn.Module):
    def __init__(self, enc_hidden_size: int, enc_bidirectional: bool, dec_hidden_size: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hidden_size * (2 if enc_bidirectional else 1) + dec_hidden_size, dec_hidden_size)
        self.energy_weights_vec = nn.Linear(dec_hidden_size, 1, bias=False)
        self.attn_weights = None

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor):
        """
        @param dec_hidden:  [batch_size, dec_hidden_size]
        @param enc_outputs: [batch_size, seq_size, hidden_size * 2]
        @return:
        """
        batch_size, seq_size = enc_outputs.shape[0], enc_outputs.shape[1]
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, seq_size, 1)
        # repeat the hidden dim seq_size times so dec_hidden -> [batch_size, seq_size, dec_hidden_size]
        cat_hiddens = torch.cat((enc_outputs, dec_hidden), dim=2)
        # catting the hiddens so -> [batch_size, seq_size, dec_hidden_size + num_directions * enc_hidden_size]
        energy = torch.tanh(self.attn(cat_hiddens))  # [batch_size, seq_size, dec_hidden_size]
        energy = self.energy_weights_vec(energy).squeeze(2)  # [batch_size, seq_size]
        self.attn_weights = F.softmax(energy, dim=1)
        return self.attn_weights


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, output_size: int, enc_hidden_size: int,
                 attention: Attention, enc_bidirectional: bool = True):
        super(Decoder, self).__init__()
        self.input_size = output_size  # = output size because we loop with input_(t) = output_(t-1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.enc_hidden_size = enc_hidden_size
        self.attention = attention
        self.embedding = nn.Linear(self.input_size, embedding_size)
        enc_num_dirs = 2 if enc_bidirectional else 1
        self.rnn = nn.GRU((enc_hidden_size * enc_num_dirs) + embedding_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_size * enc_num_dirs + embedding_size + hidden_size, output_size)

    def forward(self, dec_input: torch.Tensor, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor):
        """
        @param dec_input: [batch size, 1, enc_output_size]
        @param dec_hidden:  [batch size, dec_hidden_size]
        @param enc_outputs: [batch_size, seq_size , enc_hidden_dim * num_directions]
        @return:
        """
        embedded = self.embedding(dec_input).unsqueeze(1)
        attn_weights = self.attention(dec_hidden, enc_outputs).unsqueeze(1)  # [batch size,1, seq_size]
        weighted_enc_outputs = torch.bmm(attn_weights, enc_outputs)  # [batch size, 1, enc hidden dim * num directions]
        rnn_input = torch.cat((embedded, weighted_enc_outputs), dim=2)  # [batch size, 1,enc hid dim * dirs + emb size]
        dec_output, dec_hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(1).permute(1, 0, 2))
        # output = [batch size, seq len, dec hidden size * n directions] = [batch_size, 1, dec hidden size]
        # hidden = [n layers * n directions, batch size, dec hid dim] = [1, batch_size, dec hidden size]
        # this also means that output == hidden (up to permute)
        prediction = self.fc_out(torch.cat((dec_output, weighted_enc_outputs, embedded), dim=2)).squeeze(
            1)  # [batch size, output dim]
        return prediction, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, input_dim: Tuple, target_lag,
                 enc_embedding_size, enc_hidden_size, enc_num_layers, enc_bidirectional,
                 dec_embedding_size, dec_hidden_size, dec_output_size):
        """

        @param input_dim: [batch_size, feature_lags, encoder_input_size]
        @param target_lag: horizon to forcast
        @param enc_embedding_size: the embedding size for the encoder
        @param enc_hidden_size: the hidden size for the encoder
        @param enc_num_layers: the number of rnn cells in the encoder
        @param enc_bidirectional: True iff bidirectional
        @param dec_embedding_size: the embedding size for the decoder
        @param dec_hidden_size: the hidden size for the decoder
        @param dec_output_size: the output size for the decoder
        """
        super(Seq2Seq, self).__init__()
        self.batch_size, self.feature_lags, self.input_size = input_dim
        self.target_lag = target_lag
        self.cast_input_to_dec_output = nn.Linear(input_dim[-1], dec_output_size)
        self.encoder = Encoder(self.input_size, enc_embedding_size, enc_hidden_size, dec_hidden_size, enc_num_layers,
                               enc_bidirectional)
        self.attention = Attention(enc_hidden_size, enc_bidirectional, dec_hidden_size)
        self.decoder = Decoder(dec_embedding_size, dec_hidden_size, dec_output_size, enc_hidden_size,
                               self.attention, enc_bidirectional)
        self.output_size = self.batch_size, self.target_lag, dec_output_size

    @property
    def name(self):
        return 'seq2seq'

    def forward(self, x: torch.Tensor):
        """
        @param x: [batch_size, src len, input size]
        @return:
        """
        outputs = []
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(x)
        input = self.cast_input_to_dec_output(x[:, -1, :])
        for t in range(self.target_lag):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token
            outputs.append(output)
            input = output
        return torch.stack(outputs).to(x.device).permute(1, 0, 2)


if __name__ == '__main__':
    target_lag = 1
    enc_embedding_size = 10
    enc_hidden_size = 16
    enc_num_layers = 1
    enc_bidirectional = True
    dec_embedding_size = 10
    dec_hidden_size = 12
    dec_output_size = 1
    batch_size = 2048
    feature_lag = 480
    input_size = 7
    input_dim = batch_size, feature_lag, input_size
    xx = torch.randn(batch_size, feature_lag, input_size)
    s2s = Seq2Seq(input_dim, target_lag, enc_embedding_size, enc_hidden_size, enc_num_layers, enc_bidirectional,
                  dec_embedding_size, dec_hidden_size, dec_output_size)
    y = s2s(xx)
    print(y)
