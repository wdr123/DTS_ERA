import torch
import torch.nn as nn
import torch.nn.functional as F


import collections
class ASD_TD_CNP(nn.Module):
    '''
    The encoder
    '''

    def __init__(self,is_touch=True,is_gaze=True):
        '''
        ANP Encoder
        Can realize the CNP with identity attention
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(ASD_TD_CNP, self).__init__()

        # self.touch_lstm = nn.LSTM(3,32)
        # self.gaze_lstm = nn.LSTM(6,32)

        self._is_touch = is_touch
        self._is_gaze = is_gaze

        latent_dim = 32
        decoder_input = 64

        touch_inp= 3
        gaze_inp = 4
        if not is_touch or not is_gaze:
            decoder_input = 32

        self.touch_encoder = nn.Sequential(
            collections.OrderedDict([
                ("touchlin1",nn.Linear(touch_inp, latent_dim)),
                ("touchrelu1",nn.ReLU() ),
                ("touchlin2", nn.Linear(latent_dim, latent_dim)),
            ])
        )
        self.gaze_encoder = nn.Sequential(
            collections.OrderedDict([
                ("gazelin1",nn.Linear(gaze_inp, latent_dim)),
                ("gazerelu1",nn.ReLU() ),
                ("gazelin2", nn.Linear(latent_dim,latent_dim)),
            ])
        )

        self.decoder = nn.Sequential(
            collections.OrderedDict([
                ("declin1",nn.Linear(decoder_input, latent_dim)),
                ("decrelu1",nn.ReLU() ),
                ("decop", nn.Linear(latent_dim,1)),
            ])
        )




    def forward(self, touch_d, gaze_d, add_data=0):

        bs, num_ts, touch_dim = touch_d.shape
        _,_, gaze_dim = gaze_d.shape

        touch_d = touch_d.view(bs*num_ts,touch_dim)
        gaze_d = gaze_d.view(bs*num_ts,gaze_dim)

        # Get the shape of input
        touch_emb = self.touch_encoder(touch_d)
        gaze_emb = self.gaze_encoder(gaze_d)
        # print("first: ", touch_emb.shape)
        touch_emb = touch_emb.view(bs,num_ts,touch_emb.shape[-1])
        gaze_emb = gaze_emb.view(bs,num_ts,gaze_emb.shape[-1])


        touch_emb = torch.mean(touch_emb, dim = 1)
        gaze_emb = torch.mean(gaze_emb, dim = 1)
        # print("aggregate: ", touch_emb.shape, gaze_emb.shape)
        if not self._is_gaze:
            task_rep = touch_emb
        elif not self._is_touch:
            task_rep = gaze_emb
        else:
            task_rep = torch.cat([touch_emb,gaze_emb], axis  = -1)
        # print("task rep shape: ", task_rep.shape)
        # print('tr: ', task_rep.shape)
        # task_rep = torch.cat([task_rep, add_data], axis = -1)

        pred = self.decoder(task_rep)
        # print('pred: ', pred)
        # import sys
        # sys.exit()

        return pred

import torch
import torch.nn as nn
import torch.nn.functional as F



class ASD_TD_CNP_LSTM(nn.Module):
    '''
    The encoder
    '''

    def __init__(self,is_touch=True,is_gaze=True):
        '''
        ANP Encoder
        Can realize the CNP with identity attention
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(ASD_TD_CNP_LSTM, self).__init__()

        self._is_touch = is_touch
        self._is_gaze = is_gaze

        latent_dim = 32
        decoder_input = 64

        gaze_input_dim = 4
        touch_input_dim = 3
        if not is_touch or not is_gaze:
            decoder_input = 32

        self.touch_lstm = nn.LSTM(input_size=touch_input_dim, hidden_size=latent_dim, num_layers=1, batch_first=True)
        self.gaze_lstm = nn.LSTM(input_size=gaze_input_dim, hidden_size=latent_dim, num_layers=1, batch_first=True)

        self.touch_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.gaze_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,1),
        )




    def forward(self, touch_d, gaze_d, add_data = 0):

        bs, num_ts, touch_dim = touch_d.shape
        _, _, gaze_dim = gaze_d.shape

        # touch_d = touch_d.view(bs * num_ts, touch_dim)
        # gaze_d = gaze_d.view(bs * num_ts, gaze_dim)
        # print("input: ", touch_d.shape, gaze_d.shape)
        touch_output, touch_status = self.touch_lstm(touch_d)
        gaze_output, gaze_status = self.gaze_lstm(gaze_d)
        # print("op: ", touch_output.shape)
        # Get the shape of input
        touch_emb = self.touch_encoder(touch_output)
        gaze_emb = self.gaze_encoder(gaze_output)

        # print("first: ", touch_emb.shape)
        touch_emb = touch_emb.view(bs, num_ts, touch_emb.shape[-1])
        gaze_emb = gaze_emb.view(bs, num_ts, gaze_emb.shape[-1])

        touch_emb = torch.mean(touch_emb, dim = 1)
        gaze_emb = torch.mean(gaze_emb, dim = 1)

        if not self._is_gaze:
            task_rep = touch_emb
        elif not self._is_touch:
            task_rep = gaze_emb
        else:
            task_rep = torch.cat([touch_emb, gaze_emb], axis=-1)
        # print('tr: ', task_rep.shape)

        pred = self.decoder(task_rep)
        # print('pred: ', pred)

        return pred
