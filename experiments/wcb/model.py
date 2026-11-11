"""Plain ECG-MAE: fixed positions, global encoder/decoder, hidden waveform MSE."""
import math
import torch
from torch import nn

def positions(n,d):
    phase=torch.arange(n,dtype=torch.float32)[:,None]*torch.exp(-math.log(10000)*torch.arange(d//2)/(d/2))[None]
    return torch.cat([phase.sin(),phase.cos()],-1)

class PlainMAE(nn.Module):
    def __init__(self,p=10,size='small'):
        super().__init__(); self.p=p; self.q=2500//p; self.n=12*self.q
        e,depth,heads,d,dd,dh=(384,12,6,256,4,4) if size=='small' else (16,1,2,16,1,2)
        self.width=e
        self.patch_embed=nn.Linear(p,e)
        self.enc_lead=nn.Parameter(torch.randn(1,12,1,e)*.02)
        self.enc_time=nn.Parameter(positions(self.q,e)[None,None],requires_grad=False)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(e,heads,e*4,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True),depth,norm=nn.LayerNorm(e),enable_nested_tensor=False)
        self.to_decoder=nn.Linear(e,d); self.mask_token=nn.Parameter(torch.randn(1,1,d)*.02)
        self.dec_lead=nn.Parameter(torch.randn(1,12,1,d)*.02)
        self.dec_time=nn.Parameter(positions(self.q,d)[None,None],requires_grad=False)
        self.decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,dh,d*4,dropout=0.,activation='gelu',
            batch_first=True,norm_first=True),dd,norm=nn.LayerNorm(d),enable_nested_tensor=False)
        self.reconstruction=nn.Linear(d,p)
        for block in [*self.encoder.layers,*self.decoder.layers]: block.apply(self._initialize)
    @staticmethod
    def _initialize(m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None: nn.init.zeros_(m.in_proj_bias)
        elif isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def embedded(self,x):
        if x.shape[1:]!=(12,2500): raise ValueError('Expected B,12,2500')
        target=x.reshape(len(x),self.n,self.p)
        return target,self.patch_embed(target)+(self.enc_lead+self.enc_time).reshape(1,self.n,-1)
    def forward(self,x,mask):
        target,tokens=self.embedded(x)
        if mask.dtype!=torch.bool or mask.shape!=tokens.shape[:2]: raise ValueError('Invalid mask shape/type')
        count=(~mask).sum(-1)
        if not bool((count==self.n//2).all()): raise ValueError('Expected exact 50% globally')
        ids=torch.arange(self.n,device=x.device).expand(len(x),-1)[~mask].reshape(len(x),-1)
        visible=tokens.gather(1,ids[...,None].expand(-1,-1,tokens.shape[-1]))
        z=self.to_decoder(self.encoder(visible))
        full=self.mask_token.expand(len(x),self.n,-1).clone()
        full.scatter_(1,ids[...,None].expand(-1,-1,z.shape[-1]),z)
        full=full+(self.dec_lead+self.dec_time).reshape(1,self.n,-1)
        return self.reconstruction(self.decoder(full)),target
    def features(self,x):
        return self.encoder(self.embedded(x)[1]).mean(1)

def masked_mse(pred,target,mask):
    return ((pred-target).square().mean(-1)*mask).sum()/mask.sum()
