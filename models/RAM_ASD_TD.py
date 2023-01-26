import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import collections
import numpy as np


class GLIMPSE(nn.Module):
    '''
    The encoder
    '''

    def __init__(self, args):
        '''
        ANP Encoder
        Can realize the CNP with identity attention
        :param output_sizes: Controls the output size of the encoder representation (R)
        '''
        super(GLIMPSE, self).__init__()

        # self.touch_lstm = nn.LSTM(3,32)
        # self.gaze_lstm = nn.LSTM(6,32)

        # self._is_touch = is_touch
        # self._is_gaze = is_gaze
        self.selen = args.selen
        self.attention = args.attention
        self.model = args.model
        self.msize = args.msize

        latent_dim = args.latent
        # decoder_input = 64

        touch_gaze_inp = 7

        touch_inp = 3
        gaze_inp = 4

        self.touch_lstm = nn.LSTM(input_size=touch_inp, hidden_size=latent_dim, num_layers=1, batch_first=True)
        self.gaze_lstm = nn.LSTM(input_size=gaze_inp, hidden_size=latent_dim, num_layers=1, batch_first=True)

        # if not is_touch or not is_gaze:
        #     decoder_input = 32

        self.touch_encoder = nn.Sequential(
            collections.OrderedDict([
                # ("touchlin1", nn.Linear(touch_inp, latent_dim)),
                ("touchrelu1", nn.ReLU()),
                ("touchlin2", nn.Linear(latent_dim, latent_dim)),
            ])
        )

        self.gaze_encoder = nn.Sequential(
            collections.OrderedDict([
                # ("gazelin1", nn.Linear(gaze_inp, latent_dim)),
                ("gazerelu1", nn.ReLU()),
                ("gazelin2", nn.Linear(latent_dim, latent_dim)),
            ])
        )

        # self.decoder = nn.Sequential(
        #     collections.OrderedDict([
        #         ("declin1", nn.Linear(decoder_input, latent_dim)),
        #         ("decrelu1", nn.ReLU()),
        #         ("decop", nn.Linear(latent_dim, 1)),
        #     ])
        # )

    def forward(self, touch_d, gaze_d, l):

        bs, num_ts, touch_dim = touch_d.shape
        _, _, gaze_dim = gaze_d.shape

        # touch_d = touch_d.view(bs * num_ts, touch_dim)
        # gaze_d = gaze_d.view(bs * num_ts, gaze_dim)
        touch_output, touch_status = self.touch_lstm(touch_d)
        gaze_output, gaze_status = self.gaze_lstm(gaze_d)

        # Get the shape of input
        touch_emb = self.touch_encoder(touch_output)
        gaze_emb = self.gaze_encoder(gaze_output)
        # print("first: ", touch_emb.shape)
        touch_emb = touch_emb.view(bs, num_ts, touch_emb.shape[-1])
        gaze_emb = gaze_emb.view(bs, num_ts, gaze_emb.shape[-1])
        touch_emb_mean = torch.mean(touch_emb, dim=1)
        gaze_emb_mean = torch.mean(gaze_emb, dim=1)

        if self.model not in ['no_attention']:
            if self.attention.lower()=="sequential":
                bs_d, num_ts_d, touch_d = touch_emb.shape
                touch_d = []
                gaze_d = []
                for batch in range(bs_d):
                    if int(num_ts_d * l[batch].squeeze()) + self.selen > num_ts_d:
                        pos_begin = np.random.randint(0, num_ts_d - self.selen)
                        pos_end = pos_begin + self.selen
                        interval = [pos_begin, pos_end]
                    else:
                        interval = [int(num_ts_d * l[batch]), int(num_ts_d * l[batch]) + self.selen]

                    # print('interval', interval[0], interval[1])
                    touch_d.append(touch_emb[batch, interval[0]:interval[1], :].unsqueeze(0))
                    gaze_d.append(gaze_emb[batch, interval[0]:interval[1], :].unsqueeze(0))

                touch_emb_attn = torch.mean(torch.cat(touch_d, dim=0),dim=1) # bs*touch_emb_dim
                gaze_emb_attn = torch.mean(torch.cat(gaze_d, dim=0),dim=1) # bs*gaze_emb_dim

                if self.model == "attention_only":
                    task_rep = torch.cat([touch_emb_attn, gaze_emb_attn], axis=-1)
                else:
                    task_rep = torch.cat([touch_emb_mean, touch_emb_attn, gaze_emb_mean, gaze_emb_attn], axis=-1)

            elif self.attention.lower()=="multiple":
                bs_d, num_ts_d, touch_d = touch_emb.shape
                touch_d = []
                gaze_d = []
                attention_batch = (l*num_ts_d).type(torch.int64) # bs*msize
                for batch in range(bs_d):
                    touch_d.append(touch_emb[batch, attention_batch[batch], :].unsqueeze(0)) # 1*msize*touch_emb_dim
                    gaze_d.append(gaze_emb[batch, attention_batch[batch], :].unsqueeze(0)) # 1*msize*gaze_emb_dim
                    # print(touch_emb[batch, attention_batch[batch], :].unsqueeze(0).shape)

                touch_emb_attn = torch.mean(torch.cat(touch_d, dim=0), dim=1) # bs*touch_emb_dim
                gaze_emb_attn = torch.mean(torch.cat(gaze_d, dim=0), dim=1) # bs*gaze_emb_dim
                # print(touch_emb_attn.shape)

                if self.model == "attention_only":
                    task_rep = torch.cat([touch_emb_attn, gaze_emb_attn], axis=-1)
                else:
                    task_rep = torch.cat([touch_emb_mean, touch_emb_attn, gaze_emb_mean, gaze_emb_attn], axis=-1)

            elif self.attention.lower()=="combine":
                selen = self.selen
                bs_d, num_ts_d, touch_d = touch_emb.shape
                touch_d = []
                gaze_d = []
                touch_data = []
                gaze_data = []

                for batch in range(bs_d):
                    attention_batch = (l[batch] * num_ts_d).type(torch.int64)  # msize
                    for glimpse in range(self.msize):
                        if attention_batch[glimpse] + selen > num_ts_d:
                            pos_begin = np.random.randint(0, num_ts_d - selen)
                            pos_end = pos_begin + selen
                            interval = [pos_begin, pos_end]
                        else:
                            interval = [int(attention_batch[glimpse]), int(attention_batch[glimpse]) + selen]

                        # print(interval)

                        # print('interval', type(interval[0]), type(interval[1]))
                        touch_d.append(touch_emb[batch, interval[0]:interval[1], :].unsqueeze(0)) # 1*selen*touch_emb_dim
                        gaze_d.append(gaze_emb[batch, interval[0]:interval[1], :].unsqueeze(0)) # 1*selen*gaze_emb_dim
                        # print(touch_emb[batch, interval[0]:interval[1], :].unsqueeze(0).shape)
                    touch_data.append(torch.cat(touch_d, dim=1)) # 1,selen*msize,touch_emb_dim
                    gaze_data.append(torch.cat(gaze_d, dim=1))  # 1,selen*msize,gaze_emb_dim
                    touch_d = []
                    gaze_d = []
                touch_emb_attn = torch.mean(torch.cat(touch_data, dim=0), dim=1)  # bs,selen*msize,touch_emb_dim --> bs,touch_emb_dim
                gaze_emb_attn = torch.mean(torch.cat(gaze_data, dim=0), dim=1)  # bs,selen*msize,gaze_emb_dim --> bs, gaze_emb_dim

                if self.model == "attention_only":
                    task_rep = torch.cat([touch_emb_attn, gaze_emb_attn], axis=-1)
                else:
                    task_rep = torch.cat([touch_emb_mean, touch_emb_attn, gaze_emb_mean, gaze_emb_attn], axis=-1)

        elif self.model == "no_attention":
            task_rep = torch.cat([touch_emb_mean, gaze_emb_mean,], axis=-1)

        # print("aggregate: ", touch_emb.shape, gaze_emb.shape)
        # if not self._is_gaze:
        #     task_rep = torch.cat([touch_emb_mean,touch_emb_attn], axis=-1)
        # elif not self._is_touch:
        #     task_rep = torch.cat([gaze_emb_mean, gaze_emb_attn], axis=-1)
        # else:

        return task_rep

class CORE(nn.Module):
    '''
    Core network is a recurrent network which maintains a behavior state.
    '''
    def __init__(self, latent_dim):
        super(CORE, self).__init__()

        self.fc_h = nn.Linear(latent_dim, latent_dim)
        self.fc_g = nn.Linear(latent_dim,latent_dim)

    def forward(self, h, g):
        return F.relu(self.fc_h(h) + self.fc_g(g)) # recurrent connection
        # return F.relu(self.fc_h(h))

class LOCATION(nn.Module):
    '''
    Location network learns policy for sensing locations.
    '''
    def __init__(self, msize, latent_dim):
        super(LOCATION, self).__init__()

        self.fc = nn.Linear(latent_dim, msize)

    def forward(self, h, std):
        h = h.detach()
        l_mu = torch.tanh(self.fc(h))     # compute mean of Gaussian
        pi = Normal(l_mu, std)          # create a Gaussian distribution
        l = pi.sample()                 # sample from the Gaussian
        logpi = pi.log_prob(l)          # compute log probability of the sample
        l = torch.sigmoid(l)            # squeeze location to ensure sensing within the boundaries of an image
        return logpi, l.detach()        # logpi, l: B*2

class ACTION(nn.Module):
    '''
    Action network learn policy for task specific actions.
    In case of classification actions are possible classes.
    This network will be trained with supervised loss in case of classification.
    '''
    def __init__(self, latent_dim):
        super(ACTION, self).__init__()

        self.hidden = nn.Linear(latent_dim,32)
        self.fc = nn.Linear(32,1)

    def forward(self, h):
        hid = self.hidden(h)
        return self.fc(hid)  # Do not apply softmax as loss function will take care of it

class MODEL(nn.Module):
    '''
    Model combines all the previous elements
    '''
    def __init__(self, args):
        super(MODEL, self).__init__()
        if args.model in ["no_attention","attention_only"]:
            self.hidden = args.latent * 2
        elif args.model == "combine":
            self.hidden = args.latent * 4

        self.msize = 1
        if (args.model not in ["no_attention"]) and args.attention != "sequential":
            self.msize = args.msize
        # self.glimps = GLIMPSE(im_sz, channel, glimps_width, scale)
        self.std = args.std
        self.g = 0

        self.glimps = GLIMPSE(args)
        self.core   = CORE(self.hidden)
        self.location = LOCATION(self.msize, self.hidden)
        self.action = ACTION(self.hidden)

    def initialize(self, B, device, std):
        self.state = torch.zeros(B, self.hidden).to(device)    # initialize states of the core network
        self.l = torch.rand((B, self.msize)).to(device)   # start with a glimpse at random location
        self.std = std

    def show_loc(self,):
        return self.l

    def show_deep_set(self,):
        return self.g

    def forward(self, touch_data, gaze_data):
        # g = self.glimps(x,self.l) # glimpse encoding
        self.g = self.glimps(touch_data, gaze_data, self.l)
        self.state = self.core(self.state, self.g)         # update state of a core network based on new glimpse
        logpi, self.l = self.location(self.state, self.std)     # predict location of next glimpse
        a = self.action(self.state)                   # predict task specific actions
        return logpi, a

class LOSS(nn.Module):
    '''
    Loss function is tailored for the reward received at the end of the episode.
    Location network is trained with REINFORCE objective.
    Action network is trained with supervised objective.
    '''
    def __init__(self, T, gamma, device):
        super(LOSS, self).__init__()

        self.baseline = nn.Parameter(0.1*torch.ones(1,1).to(device), requires_grad = True) # initialize baseline to a reasonable value
        self.T = T                                                                         # length of an episode
        self.gamma = gamma # discount factor
        self.device = device

    def initialize(self, B):
        self.t = 0
        self.logpi = []

    def compute_reward(self, predict, label):
        pred = 1*(torch.sigmoid(predict)>0.5)
        return (label.detach()==pred.detach()).squeeze().float() # reward is 1 if the classification is correct and zero otherwise


    def forward(self, predict, label, logpi):
        self.t += 1
        self.logpi += [logpi]
        if self.t==self.T:
            R = self.compute_reward(predict, label)                     # reward is given at the end of the episode
            criterion = nn.BCEWithLogitsLoss(reduction='sum').to(self.device)
            a_loss = criterion(predict, label)         # supervised objective for action network
            l_loss = 0
            R_b = (R - self.baseline.detach())                      # centered rewards
            for logpi in reversed(self.logpi):
                l_loss += - (logpi.sum(-1) * R_b).sum()             # REINFORCE
                R_b = self.gamma * R_b                              # discounted centered rewards (although discount factor is always 1)
            b_loss = ((self.baseline - R)**2).sum()                 # minimize SSE between reward and the baseline
            return a_loss , l_loss , b_loss, R.sum()
        else:
            return None, None, None, None
                
def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    '''
    Decay learning rate
    '''
    if epoch in [6000,8000]:
        # lr = lr * (decay_rate ** (epoch))
        lr = lr * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

