# Copyright 2022 Fagner Cunha
# Copyright 2023 Rolnick Lab at Mila Quebec AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class ResLayer(torch.nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = torch.nn.ReLU(inplace=True)
        self.nonlin2 = torch.nn.ReLU(inplace=True)
        self.dropout1 = torch.nn.Dropout()
        self.w1 = torch.nn.Linear(self.l_size, self.l_size)
        self.w2 = torch.nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(torch.nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
        super(FCNet, self).__init__()
        num_users = 1 if num_users < 1 else num_users
        self.inc_bias = False
        self.class_emb = torch.nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = torch.nn.Linear(num_filts, num_users, bias=self.inc_bias)
        self.feats = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_filts),
            torch.nn.ReLU(inplace=True),
            ResLayer(num_filts),
            ResLayer(num_filts),
            ResLayer(num_filts),
            ResLayer(num_filts),
        )

    def forward(self, x, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb

        class_pred = self.class_emb(loc_emb)
        return torch.sigmoid(class_pred)
