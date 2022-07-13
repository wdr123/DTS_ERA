import numpy as np

from models.np_blocks import ANPDeterministicEncoder
from models.np_blocks import ANPLatentEncoder
from models.np_blocks import ANPDecoder
from models.np_template import np_model
import torch


class ANPModel(np_model):

    def __init__(self,
                 latent_encoder_output_size,
                 deterministic_encoder_output_size,
                 decoder_output_size,
                 args = None,
                 attention=None, ):
        super(ANPModel, self).__init__(args)
        if args == None:
            raise NotImplementedError
        np_model.__init__(self, args)

        if self._use_deterministic_path:  # CNP or ANP Modelx
            self._deterministic_encoder = ANPDeterministicEncoder(deterministic_encoder_output_size, attention)

        if self._use_latent_path:
            self._latent_encoder = ANPLatentEncoder(latent_encoder_output_size)

        self._decoder = ANPDecoder(decoder_output_size, args=args)
        # print("Decoder: ", self._decoder)

        # print("The NP Model")


    def forward(self, query, target_y=None):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :return:
        '''
        # print("Forward ANP")
        (context_x, context_y), target_x = query

        if self._use_latent_path:
            # print("Forward ANP Latent")
            ##PRIOR
            ctx_lat_dist, ctx_lat_mean, ctx_lat_var = self._latent_encoder(context_x,context_y)

            # During training, we have target_y. We use the target for latent encoder
            if target_y is None:
                # sample = torch.randn(ctx_lat_mean.shape).to(ctx_lat_mean.device)
                # latent_rep_sample = ctx_lat_mean + (ctx_lat_var * sample)

                latent_rep_sample = ctx_lat_dist.rsample()
            else:
                ##POSTERIOR
                tar_lat_dist, tar_lat_mean, tar_lat_var = self._latent_encoder(target_x, target_y)

                # sample = torch.randn(tar_lat_mean.shape).to(tar_lat_mean.device)
                # latent_rep_sample = tar_lat_mean + (tar_lat_var * sample)
                latent_rep_sample = tar_lat_dist.rsample()

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        dist, mu, sigma = self._decoder(representation, target_x)


        # Training tasks
        if target_y is not None:
            # print("Training tasks")

            log_likelihood = dist.log_prob(target_y)

            # print("log likelihood: ", log_likelihood.shape, "targety: ", target_y.shape)
            recons_loss = -torch.mean(log_likelihood)
            kl_loss = None
            if self._use_latent_path:
                dist_1 = torch.distributions.Normal(ctx_lat_mean, ctx_lat_var)
                dist_2 = torch.distributions.Normal(tar_lat_mean, tar_lat_var)
                kl_loss_dir = torch.distributions.kl_divergence(dist_2, dist_1)
                # kl_los =  torch.log(ctx_lat_var) -  torch.log(tar_lat_var) \
                #           + 0.5 * ( tar_lat_var**2 / ctx_lat_var**2 + \
                #                 (tar_lat_mean - ctx_lat_mean) ** 2 / ctx_lat_var**2 - 1)
                #
                # # Consider shape and think here
                # kl_loss = torch.mean(kl_los)
                # print("kl: ", torch.mean(kl_loss_dir))
                # print("cur kl: ", kl_loss)
                loss = recons_loss + torch.mean(kl_loss_dir)
            else:
                loss = recons_loss

        else:
            recons_loss = None
            kl_loss = None
            loss = None

        return dist, mu, sigma, recons_loss, kl_loss, loss

    def test_get_encoder_representation(self, query):
        '''
        :param query: ( (context_x, context_y), target_x ).
        :param target_y:
        :param epoch:
        :return:
        '''
        (context_x, context_y), target_x = query

        if self._use_latent_path:
            ##PRIOR
            ctx_lat_dist, ctx_lat_mean, ctx_lat_var = self._latent_encoder(context_x,context_y)

            sample = torch.randn(ctx_lat_mean.shape).to(ctx_lat_mean.device)
            latent_rep_sample = ctx_lat_mean + (ctx_lat_var * sample)

            batch_size, set_size, _ = target_x.shape
            latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat([1, set_size, 1])

        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)

        if self._use_deterministic_path and self._use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self._use_latent_path:
            representation = latent_rep_sample
        elif self._use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        representation = torch.mean(representation,dim=1)
        return representation



from models.np_blocks import ANPEvidentialDecoder

from models.np_blocks import ANPEvidentialLatentEncoder

#############Attentive Models with self attention
from models.attention_model import *
class ANP_LatentModel(nn.Module):
    """
        Latent Model (Attentive Neural Process)
        Fixed Multihead Attention
        """

    def __init__(self, latent_encoder_sizes,
                 determministic_encoder_sizes,
                 decoder_output_size,
                 args,
                 attention,):
        super(ANP_LatentModel, self).__init__()
        num_hidden = latent_encoder_sizes[1]
        self.args = args

        self.latent_encoder = LatentEncoder(num_hidden, num_hidden,input_dim=latent_encoder_sizes[0])
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden,input_dim=determministic_encoder_sizes[0])

        self.decoder = ANPDecoder(decoder_output_size, args=args)

    def forward(self,query, target_y=None):

        (context_x, context_y), target_x = query

        num_targets = target_x.size(1)

        if self.args.use_latent_path:
            ctx_lat_dist, ctx_lat_mu, ctx_lat_std = self.latent_encoder(context_x, context_y)

            if target_y is None:
                latent_rep_sample = ctx_lat_dist.rsample()
            else:
                tar_lat_dist, tar_lat_mu, tar_lat_std = self.latent_encoder(target_x, target_y)
                latent_rep_sample = tar_lat_dist.rsample()


        latent_rep_sample = latent_rep_sample.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self.args.use_deterministic_path:
            deterministic_rep = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        if self.args.use_deterministic_path and self.args.use_latent_path:
            representation = torch.cat((deterministic_rep, latent_rep_sample), dim=-1)
        elif self.args.use_latent_path:
            representation = latent_rep_sample
        elif self.args.use_deterministic_path:
            representation = deterministic_rep
        else:
            raise ValueError("You need at least one path for the encoder")

        dist, mu, std = self._decoder(representation, target_x)

        # For Training
        if target_y is not None:
            # get log probability
            log_likelihood = dist.log_prob(target_y)

            kl_loss = None
            loss = torch.zeros(size=(1,), device=target_y.device)

            # loss += torch.mean(0.5*(target_y-mu)**2)#-torch.mean(log_likelihood)
            recons_loss = -torch.mean(log_likelihood)
            loss += recons_loss

            # get KL divergence between prior and posterior
            if self.args.use_latent_path:
                dist_1 = torch.distributions.Normal(ctx_lat_mu, ctx_lat_std)
                dist_2 = torch.distributions.Normal(tar_lat_mu, tar_lat_std)
                kl_loss = torch.distributions.kl_divergence(dist_2, dist_1)

                loss += torch.mean(kl_loss)

        else:
            recons_loss = None
            kl_loss = None
            loss = None

        return dist, mu, std, recons_loss, kl_loss, loss



