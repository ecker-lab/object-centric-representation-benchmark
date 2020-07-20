import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrb.vimon.utils import Resize, Flatten


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resize=True, factor=2., skip=True):
        super().__init__()

        self._factor = factor
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.inst_norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self._resize = resize
        self._skip = skip
        
    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        out = skip = self.relu(x)
        
        if self._resize:    
            out = F.interpolate(skip, scale_factor=self._factor, mode='nearest')

        if self._skip:
            return out, skip
        else:
            return out


class Encoder(nn.Module):
    def __init__(self, block, in_ch, out_ch):
        super().__init__()
        assert len(out_ch) == len(in_ch)
        
        self.n_blocks = len(out_ch)
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            resize = False if i == self.n_blocks -1 else True
            self.blocks.append(block(in_ch[i], out_ch[i], resize=resize, factor=.5))

    def forward(self, x):
        skips = []
        for i in range(self.n_blocks):
            x, skip = self.blocks[i](x)
            skips.append(skip)  
        return x, skips
    

class Decoder(nn.Module):
    def __init__(self, block, in_ch, out_ch):
        super().__init__()
        assert len(out_ch) == len(in_ch)
        
        self.n_blocks = len(out_ch)
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            resize = False if i == self.n_blocks -1 else True
            self.blocks.append(block(in_ch[i], out_ch[i], resize=resize, factor=2., skip=False))

        self.conv = nn.Conv2d(out_ch[-1], 1, 1, 1, 0, bias=False)

    def forward(self, x, skips):
        for i in range(self.n_blocks):
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = self.blocks[i](x)
        x = self.conv(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, encoder, decoder, in_ch=4, mlp_ch=256, n_sp=4, n_ch=4096):
        super(AttentionModule, self).__init__()
        self.in_ch = in_ch      
        self.encoder = encoder
        self.decoder = decoder
        self.mlp = nn.Sequential(
                        Flatten(),
                        nn.Linear(n_ch, 128),
                        nn.ReLU(True),
                        nn.Linear(128, 128),
                        nn.ReLU(True),
                        nn.Linear(128, n_ch),
                        nn.ReLU(True),
                        Resize((-1, mlp_ch, n_sp, n_sp)))

    def forward(self, x):                      
        assert x.size(1) == self.in_ch                                 
        x, skips = self.encoder(x)
        x = self.mlp(x)
        x = self.decoder(x, skips)
        return x

    
def create_attn_model(in_ch=4):
    """Constructs attention network for ViMON.
    """
    encoder = Encoder(AttentionBlock, [in_ch, 32, 32, 32, 32], [32, 32, 32, 32, 32])
    decoder = Decoder(AttentionBlock, [64, 64, 64, 64, 64], [32, 32, 32, 32, 32])
                              
    model = AttentionModule(encoder, decoder, in_ch, mlp_ch=32, n_ch=512, n_sp=4)
    return model
