import torch
from torch import nn
import torch.nn.functional as F
from pytorch_tcn import TCN


# --ST-GRU--#
class ST_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(ST_GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.FC_layer = nn.Linear(hidden_size, 3)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        h0 = h0.cuda() if torch.cuda.is_available() else h0.cpu()
        h, _ = self.gru(X, h0.detach())
        y_preds = self.FC_layer(h)
        return y_preds

    def predict(model, X_train, X_test, output_steps):
        model = model.eval()
        X_test = X_test.view(-1, output_steps, X_train.size(1))
        return model.forward(X_test)


# --T-GRU--#
class T_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(T_GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.FC_layer = nn.Linear(hidden_size, 3)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        h0 = h0.cuda() if torch.cuda.is_available() else h0.cpu()
        h, _ = self.gru(X, h0.detach())
        y_preds = self.FC_layer(h)
        return y_preds

    def predict(model, X_train, X_test, output_steps):
        model = model.eval()
        X_test = X_test.view(-1, output_steps, X_train.size(1))
        return model.forward(X_test)


# --S-DNN--#
class S_DNN(nn.Module):
    def __init__(self, size, batch_norm=False, dropout_rate=0.3):
        super(S_DNN, self).__init__()
        self.size = size
        self.batch_norm = batch_norm
        self.batch_norm_input = (
            nn.BatchNorm1d(size[0], momentum=0.5) if self.batch_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.regression_head = nn.Linear(size[-2], size[-1])
        self.linear_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList() if self.batch_norm else None

        for i in range(len(size) - 2):
            lin_layer = nn.Linear(size[i], size[i + 1])
            self._set_ini(lin_layer)
            self.linear_layers.append(lin_layer)

            if self.batch_norm:
                batch_norm_layer = nn.BatchNorm1d(size[i + 1], momentum=0.99)
                self.batch_norm_layers.append(batch_norm_layer)

    def _set_ini(self, layer):
        nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        y = self.batch_norm_input(x)

        for i, lin_layer in enumerate(self.linear_layers):
            y = lin_layer(y)
            if self.batch_norm:
                y = self.batch_norm_layers[i](y)
            y = F.relu(y)
            if i == 0:
                y = self.dropout(y)

        y_preds = self.regression_head(y)
        return y_preds


# --TCN--#
"""
Using pytorch implementation from https://github.com/paul-krug/pytorch-tcn #
"""

# --ST-LSTM--#
"""
Implementation inspired by Youtuber Greg Hogg's LSTM tutorial
Link: https://www.youtube.com/watch?v=q_HS4s1L8UI&ab_channel=GregHogg
"""


class ST_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(ST_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, out_features=3)  # 3D coords

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).cuda()
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out
