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

        self._is_touch = is_touch
        self._is_gaze = is_gaze

        self.num_inst_pts = 25

        latent_dim = 32
        decoder_input = 64

        touch_inp= 4
        gaze_inp = 5
        if not is_touch or not is_gaze:
            decoder_input = 32

        self.touch_encoder = nn.Sequential(
            collections.OrderedDict([
                ("touchlin1",nn.Linear(touch_inp, latent_dim)),
                ("touchrelu1",nn.ReLU() ),
                ("touchlin2", nn.Linear(latent_dim,latent_dim)),
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




    def forward(self, touch_d_asd, gaze_d_asd, touch_d_td, gaze_d_td):


        bs, num_ts_asd, touch_dim_asd = touch_d_asd.shape
        _,_, gaze_dim_asd = gaze_d_asd.shape

        bs, num_ts_td, touch_dim_td = touch_d_td.shape
        _,_, gaze_dim_td = gaze_d_td.shape

        touch_d_asd = touch_d_asd.view(bs*num_ts_asd,touch_dim_asd)
        gaze_d_asd = gaze_d_asd.view(bs*num_ts_asd,gaze_dim_asd)

        touch_d_td = touch_d_td.view(bs*num_ts_td,touch_dim_td)
        gaze_d_td = gaze_d_td.view(bs*num_ts_td,gaze_dim_td)

        # Get the shape of input
        touch_emb_asd = self.touch_encoder(touch_d_asd)
        gaze_emb_asd = self.gaze_encoder(gaze_d_asd)

        touch_emb_td = self.touch_encoder(touch_d_td)
        gaze_emb_td = self.gaze_encoder(gaze_d_td)



        # print("first: ", touch_emb.shape)
        touch_emb_asd = touch_emb_asd.view(-1,self.num_inst_pts,touch_emb_asd.shape[-1])
        gaze_emb_asd = gaze_emb_asd.view(-1,self.num_inst_pts,gaze_emb_asd.shape[-1])

        touch_emb_td = touch_emb_td.view(-1,self.num_inst_pts,touch_emb_td.shape[-1])
        gaze_emb_td = gaze_emb_td.view(-1,self.num_inst_pts,gaze_emb_td.shape[-1])


        touch_emb_asd = torch.mean(touch_emb_asd, dim = 1)
        gaze_emb_asd = torch.mean(gaze_emb_asd, dim = 1)

        touch_emb_td = torch.mean(touch_emb_td, dim = 1)
        gaze_emb_td = torch.mean(gaze_emb_td, dim = 1)
        if not self._is_gaze:
            task_rep_asd = touch_emb_asd
            task_rep_td = touch_emb_td
        elif not self._is_touch:
            task_rep_asd = gaze_emb_asd
            task_rep_td = gaze_emb_td
        else:
            task_rep_asd = torch.cat([touch_emb_asd,gaze_emb_asd], axis  = -1)
            task_rep_td = torch.cat([touch_emb_td,gaze_emb_td], axis  = -1)

        pred_asd = self.decoder(task_rep_asd)
        pred_td = self.decoder(task_rep_td)

        pred_asd = torch.sigmoid(pred_asd)
        pred_td = torch.sigmoid(pred_td)

        return pred_asd, pred_td

    def forward_test(self, touch_d, gaze_d):

        bs, num_ts_asd, touch_dim_asd = touch_d.shape
        _, _, gaze_dim_asd = gaze_d.shape


        touch_d = touch_d.view(bs * num_ts_asd, touch_dim_asd)
        gaze_d = gaze_d.view(bs * num_ts_asd, gaze_dim_asd)


        # Get the shape of input
        touch_emb_asd = self.touch_encoder(touch_d)
        gaze_emb_asd = self.gaze_encoder(gaze_d)



        # print("first: ", touch_emb.shape)
        touch_emb_asd = touch_emb_asd.view(-1, self.num_inst_pts, touch_emb_asd.shape[-1])
        gaze_emb_asd = gaze_emb_asd.view(-1, self.num_inst_pts, gaze_emb_asd.shape[-1])

        touch_emb_asd = torch.mean(touch_emb_asd, dim=1)
        gaze_emb_asd = torch.mean(gaze_emb_asd, dim=1)
        # print("aggregate: ", touch_emb.shape, gaze_emb.shape)
        if not self._is_gaze:
            task_rep_asd = touch_emb_asd
        elif not self._is_touch:
            task_rep_asd = gaze_emb_asd
        else:
            task_rep_asd = torch.cat([touch_emb_asd, gaze_emb_asd], axis=-1)

        pred_asd = self.decoder(task_rep_asd)
        pred_asd = torch.sigmoid(pred_asd)

        return pred_asd


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
