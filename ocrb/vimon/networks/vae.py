import torch
from torch import nn
from ocrb.vimon.utils import Resize


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['vae']['latent_dim']

        self.f = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
            Resize((-1, 1024)),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 2*self.latent_dim)
            )

    def forward(self, x):
        mu_logvar = self.f(x)
        return mu_logvar
    
    
class BroadcastConvDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.im_size = config['data']['im_size'] + 8
        self.latent_dim = config['vae']['latent_dim']
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(self.latent_dim+2, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 4, 1, 1, 0)
                    )

    def init_grid(self):
        x = torch.linspace(-1, 1, self.im_size)
        y = torch.linspace(-1, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        
        
    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z)
        x_k_mu = x[:, :3]
        m_k_logits = x[:, 3:]        
        return x_k_mu, m_k_logits
