#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/28 14:03
# @Author  : 马赫
# @Email   : 1692303843@qq.com
# @FileName: model.py


import warnings

warnings.filterwarnings("ignore")

import paddle
from paddle import nn
from paddle.nn.utils import weight_norm


# import paddlenlp as nlp


# 单LSTM网络
class LSTM(paddle.nn.Layer):
    def __init__(self, input_channels=7, output_channels=2):
        super(LSTM, self).__init__()
        self.lstm = paddle.nn.LSTM(input_size=input_channels,
                                   hidden_size=50,
                                   num_layers=4,
                                   dropout=0.2,
                                   time_major=False)  # 要求输入的形状是[batch_size,time_steps,input_size]
        self.fc = paddle.nn.Linear(in_features=50, out_features=output_channels)

    def forward(self, inputs):
        outputs, final_states = self.lstm(inputs)  # 使用最后一层的最后一个step的输出作为线性层的输入
        y = self.fc(final_states[0][3])  # 输入：形状为 [batch_size,∗,in_features] 的多维Tensor。
        y = paddle.unsqueeze(y, [1])
        return y


# paddle.summary(LSTM(11, 4), (256, 187, 11))


# 三层LSTM模型，效果差
class TriLSTM(nn.Layer):
    def __init__(self, input_channels=7, output_channels=2):
        super(TriLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=128, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=output_channels, num_layers=1)
        self.fc = nn.Linear(187, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = self.fc(x)
        x = paddle.transpose(x, (0, 2, 1))
        # print(x.shape)

        return x


# print('=' * 40)
# paddle.summary(TriLSTM(11, 4), (256, 187, 11))


# TCN模型
class Chomp1d(nn.Layer):
    """
    Remove the elements on the right.
    Args:
        chomp_size (int): The number of elements removed.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Layer):
    """
    The TCN block, consists of dilated causal conv, relu and residual block.
    See the Figure 1(b) in https://arxiv.org/pdf/1803.01271.pdf for more details.
    Args:
        n_inputs ([int]): The number of channels in the input tensor.
        n_outputs ([int]): The number of filters.
        kernel_size ([int]): The filter size.
        stride ([int]): The stride size.
        dilation ([int]): The dilation size.
        padding ([int]): The size of zeros to be padded.
        dropout (float, optional): Probability of dropout the units. Defaults to 0.2.
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1D(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        # Chomp1d is used to make sure the network is causal.
        # We pad by (k-1)*d on the two sides of the input for convolution,
        # and then use Chomp1d to remove the (k-1)*d output elements on the right.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1D(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1D(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv1.weight.shape))
        self.conv2.weight.set_value(
            paddle.tensor.normal(0.0, 0.01, self.conv2.weight.shape))
        if self.downsample is not None:
            self.downsample.weight.set_value(
                paddle.tensor.normal(0.0, 0.01, self.downsample.weight.shape))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Layer):
    """
    A `TCNEncoder` takes as input a sequence of vectors and returns a
    single vector, which is the last one time step in the feature map.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
    and the output is of shape `(batch_size, num_channels[-1])` with a receptive
    filed:

    .. math::

        receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).

    Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
    such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.
    Args:
        input_size (int): The number of expected features in the input (the last dimension).
        num_channels (list): The number of channels in different layer.
        kernel_size (int): The kernel size. Defaults to 2.
        dropout (float): The dropout probability. Defaults to 0.2.
    """

    def __init__(self, input_size, num_channels, time_steps=1, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self._input_size = input_size
        self._output_dim = num_channels[-1]
        self.time_steps = time_steps
        layers = nn.LayerList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout))

        self.network = nn.Sequential(*layers)

    def get_input_dim(self):
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `TCNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self._input_size

    def get_output_dim(self):
        """
        Returns the dimension of the final vector output by this `TCNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self._output_dim

    def forward(self, inputs):
        """
        TCNEncoder takes as input a sequence of vectors and returns a
        single vector, which is the last one time step in the feature map.
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
        and the output is of shape `(batch_size, num_channels[-1])` with a receptive
        filed:

        .. math::

            receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).
        Args:
            inputs (Tensor): The input tensor with shape `[batch_size, num_tokens, input_size]`.
        Returns:
            Tensor: Returns tensor `output` with shape `[batch_size, num_channels[-1]]`.
        """
        inputs_t = inputs.transpose([0, 2, 1])
        # output = self.network(inputs_t).transpose([2, 0, 1])[-1]
        output1 = self.network(inputs_t)
        output2 = output1.transpose([0, 2, 1])
        output = output2[:, -self.time_steps:, :]

        return output


class TimeSeriesNetwork(nn.Layer):
    def __init__(self, in_features, out_features=1, time_steps=1, num_channels=[256]):
        super(TimeSeriesNetwork, self).__init__()
        self.last_num_channel = num_channels[-1]
        self.tcn = TCNEncoder(in_features, num_channels, time_steps,
                              kernel_size=3, dropout=0.2)
        self.linear = nn.Linear(in_features=self.last_num_channel, out_features=out_features)

    def forward(self, x):
        tcn_out = self.tcn(x)
        y_pred = self.linear(tcn_out)
        return y_pred


# TCNEncoder 输入形状(batch_size, time_steps, features) 输出形状(batch_size, ?)
# paddle.summary(TimeSeriesNetwork(11, 4, time_steps=1, num_channels=[256, ]), (32, 187, 11))


class MPFNet(nn.Layer):
    def __init__(self, in_features, out_features=1, time_steps=1, num_channels=[256], hidden_channels=32):
        super(MPFNet, self).__init__()

        self.tcn1 = TCNEncoder(in_features, num_channels, time_steps, kernel_size=3, dropout=0.2)
        # self.conv1 = nn.Conv1D(num_channels[0], out_features, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(num_channels[0], out_features)

        self.tcn2 = TCNEncoder(num_channels[0], num_channels, time_steps, kernel_size=3, dropout=0.2)
        # self.conv2 = nn.Conv1D(num_channels[0], out_features, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(num_channels[0], out_features)

        self.tcn3 = TCNEncoder(num_channels[0], num_channels, time_steps, kernel_size=3, dropout=0.2)
        # self.conv3 = nn.Conv1D(num_channels[0], out_features, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(num_channels[0], out_features)

        # self.conv = nn.Conv1D(3*time_steps, time_steps, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(3 * time_steps, time_steps)

    def Time_Decoder(self, layer, x):
        x = x.transpose([0, 2, 1])
        x = layer(x)
        x = x.transpose([0, 2, 1])
        return x

    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        # t1_dec = self.Time_Decoder(self.conv1, tcn_out1)
        t1_dec = self.fc1(tcn_out1)

        tcn_out2 = self.tcn2(tcn_out1)
        # t2_dec = self.Time_Decoder(self.conv2, tcn_out2)
        t2_dec = self.fc2(tcn_out2)

        tcn_out3 = self.tcn3(tcn_out2)
        # t3_dec = self.Time_Decoder(self.conv3, tcn_out3)
        t3_dec = self.fc2(tcn_out3)

        # 对时间维度堆叠合并，并一维卷积
        t_dec = paddle.concat([t1_dec, t2_dec, t3_dec], axis=1)
        # y_pred = self.conv(t_dec)
        y_pred = self.Time_Decoder(self.fc, t_dec)

        return y_pred


class PositionalEncoding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)

        div_term = paddle.arange(0, d_model, 2, dtype='float32')
        div_term = paddle.exp(-paddle.log(paddle.to_tensor(10000.0)) / d_model * div_term)
        div_term1 = paddle.arange(1, d_model, 2, dtype='float32')
        div_term1 = paddle.exp(-paddle.log(paddle.to_tensor(10000.0)) / d_model * div_term1)

        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term1)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe, persistable=True)

    def forward(self, x):
        return x + self.pe[: x.shape[0], :]


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class Transformer(nn.Layer):
    def __init__(self, input_channels, output_channels=1, time_steps=1, nhead=10, dim_feedforward=512, num_layers=4,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.time_steps = time_steps
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_channels, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        initrange = 0.1
        self.decoder = nn.Linear(
            input_channels, output_channels,
            weight_attr=paddle.nn.initializer.Uniform(-initrange, initrange),
            bias_attr=paddle.nn.initializer.Constant()
        )

    def _generate_square_subsequent_mask(self, length):
        mask = (paddle.triu(paddle.ones([length, length])) == 1).T.astype('float')
        mask = masked_fill(x=mask, mask=mask == 0, value=float('-inf'))
        mask = masked_fill(x=mask, mask=mask == 1, value=float('0.0'))
        return mask

    def forward(self, src):
        src_len = src.shape[1]
        if self.src_mask is None or self.src_mask.shape[1] != src_len:
            mask = self._generate_square_subsequent_mask(src_len)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output[:, -self.time_steps:, :]

        output = self.decoder(output)
        return output


# Transformer 输入形状(batch_size, time_steps, features) 输出形状(batch_size, ?)
# paddle.summary(Transformer(15, 4, nhead=3), (12, 187, 15))


class Net(nn.Layer):
    def __init__(self, input_channels=23, output_channels=4, time_steps=1, model='TCN'):
        super(Net, self).__init__()
        if model == 'LSTM':
            self.net = LSTM(input_channels, output_channels)
        elif model == 'TriLSTM':
            self.net = TriLSTM(input_channels, output_channels)
        elif model == 'Transformer':
            self.net = Transformer(input_channels, output_channels, time_steps, nhead=1)
        elif model == 'TCN':
            self.net = TimeSeriesNetwork(input_channels, output_channels, time_steps, num_channels=[256])
        elif model == 'MPFNet':
            self.net = MPFNet(input_channels, output_channels, time_steps, num_channels=[256], hidden_channels=32)

        else:
            raise Exception('Choose Model')

    def forward(self, x):
        y = self.net(x)
        return y
