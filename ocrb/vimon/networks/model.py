import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrb.vimon.networks.vae import ConvEncoder, BroadcastConvDecoder
from ocrb.vimon.networks.attention import create_attn_model


class ViMON(nn.Module):
    def __init__(self, config, attn_module, vae_encoder, vae_decoder, inference=False):
        super().__init__()

        self.inference = inference
        self.attn_module = attn_module
        self.n_slots = config['model']['n_slots']
        self.n_steps = config['data']['n_steps'] if inference else config['data']['n_steps'] - 1
        self.latent_dim = config['vae']['latent_dim']
        self.gru_dim = config['gru']['latent_dim']
        self.sigma = torch.cat([torch.Tensor([0.09]), torch.Tensor([0.11]).repeat(self.n_slots-1)])[None, :, None, None, None]
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        self.gru = nn.GRU(self.latent_dim*2, self.gru_dim, batch_first=True)
        self.mlp = nn.Linear(self.gru_dim, 2*self.latent_dim)
        self.linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def init_weigths(self):
        #init GRU weights
        self.gru.bias_hh_l0.data.fill_(0)
        self.gru.bias_ih_l0.data.fill_(0)
        self.gru.weight_hh_l0.data.normal_(0, 1e-3)
        rand = torch.randn(3*self.gru_dim, 2*self.latent_dim) * 1e-3
        eye = rand + torch.cat([torch.zeros(2*self.gru_dim, 2*self.latent_dim), torch.eye(self.gru_dim)[:,:2*self.latent_dim]]) 
        self.gru.weight_ih_l0.data.copy_(eye)
        #init linear layer weights
        self.mlp.apply(self.init_eye)
        self.linear.apply(self.init_eye)
        
    def init_eye(self, m):
        if type(m) == nn.Linear:
            m.bias.data.fill_(0)
            torch.nn.init.eye_(m.weight)
        
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod   
    def gauss_prob(preds, targets, sigma):
        return torch.exp(-torch.pow(preds - targets, 2) / ( 2 * torch.pow(sigma, 2))) / torch.sqrt(2 * sigma**2 * math.pi)

    @staticmethod
    def kl_div(mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=3).mean(dim=(0, 1)).sum()

    def decoder_loglikelihood(self, target, slot_recon, mask_recon):
        b, T, _, _, _, _ = target.size()
        mask_recon = F.softmax(mask_recon, 2)
        x_prob = self.gauss_prob(slot_recon, target, self.sigma.to(slot_recon.device))
        masked_x_prob = torch.clamp(torch.sum(mask_recon * x_prob, dim=2), min=1e-5)
        return - torch.log(masked_x_prob).sum() / (b * T)

    def accumulate_losses(self, target, curr_slots, curr_masks, next_slots, next_masks, attn_masks):
        b, T, _, _, _ = target.size()
        # decoder loglikelihoods
        ## reconstruction
        recon_loss = self.decoder_loglikelihood(target[:, :-1, None, ...], curr_slots, curr_masks)
        # next-step prediction
        pred_loss = self.decoder_loglikelihood(target[:, 2:, None, ...], next_slots[:, 1:], next_masks[:, 1:])
        
        # mask losses
        ## reconstruction
        curr_mask_loss = self.criterion_kl(curr_masks.log_softmax(dim=2), attn_masks) / (T-1)
        # next-step prediction
        next_mask_loss = self.criterion_kl(next_masks.log_softmax(dim=2)[:, 1:-1], attn_masks[:, 2:]) / (T-2)

        losses = {
            'recon_loss': recon_loss,
            'pred_loss': pred_loss,
            'curr_mask_loss': curr_mask_loss,
            'next_mask_loss': next_mask_loss
        }
        return losses

        
    def forward(self, x):
        b, T, c, h, w = x.size()

        # create empty tensors
        attn_log_masks = torch.empty((b, self.n_steps, self.n_slots, 1, h, w), dtype=torch.float, device=x.device)

        curr_masks = torch.empty((b, self.n_steps, self.n_slots, 1, h, w), dtype=torch.float, device=x.device)
        curr_slots = torch.empty((b, self.n_steps, self.n_slots, 3, h, w), dtype=torch.float, device=x.device)

        next_masks = torch.empty((b, self.n_steps, self.n_slots, 1, h, w), dtype=torch.float, device=x.device)
        next_slots = torch.empty((b, self.n_steps, self.n_slots, 3, h, w), dtype=torch.float, device=x.device)

        mu = torch.empty((b, self.n_steps, self.n_slots, self.latent_dim), dtype=torch.float, device=x.device)
        logvar = torch.empty((b, self.n_steps, self.n_slots, self.latent_dim), dtype=torch.float, device=x.device)

        prev_hidden = torch.zeros((self.n_slots, 1, b, self.gru_dim), dtype=torch.float, device=x.device)
        prev_masks = torch.zeros((b, self.n_slots, 1, h, w), dtype=torch.float, device=x.device)
                
        for t in range(self.n_steps):
            log_s_k = torch.zeros((b, 1, h, w), dtype=torch.float, device=x.device)
            hidden = []
            masks = []
            
            if t != 0:
                prev_masks = self.logsoftmax(prev_masks)

            for k in range(self.n_slots):
                if k != self.n_slots - 1:
                    input_attn = torch.cat([x[:,t], log_s_k, prev_masks[:,k]], dim=1)
                    alpha_k = self.attn_module(input_attn)
                    # Compute attn mask
                    log_m_k = log_s_k + F.logsigmoid(alpha_k)
                    # Compute scope
                    log_s_k = log_s_k + F.logsigmoid(-alpha_k) 
                else:
                    log_m_k = log_s_k
                
                input_enc = torch.cat([x[:,t], log_m_k], dim=1)
                z_k = self.vae_encoder(input_enc)
                
                # update GRU
                z_ = z_k[:, None, :] / 5.
                out_gru_k, hidden_k = self.gru(z_, prev_hidden[k].contiguous())
                out_gru_k_ = out_gru_k * 5.
                out_k = self.mlp(out_gru_k_)
                
                mu[:, t, k] = out_k[:, 0, :self.latent_dim]
                logvar[:, t, k] = out_k[:, 0, self.latent_dim:]
                
                z_t = self.reparameterize(mu[:, t, k], logvar[:, t, k])
                z_t_pred = self.linear(z_t)

                curr_slots[:, t, k], curr_masks[:, t, k] = self.vae_decoder(z_t)
                next_slots[:, t, k], next_masks[:, t, k] = self.vae_decoder(z_t_pred)
                
                masks.append(next_masks[:, t, k])  
                attn_log_masks[:, t, k] = log_m_k
                hidden.append(hidden_k)
                
            prev_hidden = hidden
            prev_masks = torch.stack(masks, 1)
        
        attn_masks = attn_log_masks.exp()
        losses = {}
        # accumulate losses
        if not self.inference:
            kl_loss = self.kl_div(mu, logvar)
            losses = self.accumulate_losses(x, curr_slots, curr_masks, next_slots, next_masks, attn_masks)
            losses.update({'kl_loss': kl_loss})

        # accumulate recons
        curr_masks = F.softmax(curr_masks, 2)
        next_masks = F.softmax(next_masks, 2)
        recon_vae = (curr_masks.detach() * curr_slots.detach()).sum(dim=2)
        recon_attn = (attn_masks.detach() * curr_slots.detach()).sum(dim=2)
        pred_vae = (next_masks.detach() * next_slots.detach()).sum(dim=2)

        results = {
            'curr_masks': curr_masks.detach(),
            'next_masks': next_masks.detach(),
            'attn_masks': attn_masks.detach(),
            'recon_vae': recon_vae,
            'recon_attn': recon_attn,
            'pred_vae': pred_vae,
            'curr_slots':  curr_slots.detach(),
            'next_slots':  next_slots.detach(),
        }
        return results, losses


def build_vimon(config, inference=False):
    vae_encoder = ConvEncoder(config)
    vae_decoder = BroadcastConvDecoder(config)
    attn_module = create_attn_model(in_ch=5)
    vimon = ViMON(config, attn_module, vae_encoder, vae_decoder, inference=inference)
    return vimon
