import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
import random

class MalOptimizerNoise: 
    def __init__(self,score,device,γ,β,δ,lr1,nep,k,n_max):
        self.device = device
        self.score = score.requires_grad_(True).to(device)
        # self.α=α
        self.γ=γ
        self.β=β                 
        self.δ=δ
        self.lr1=lr1     
        self.nep=nep
        self.k=k
        # self.l=l
        self.n_max=n_max
        # self.n_min=n_min
                

    def optimize_noise_mask(self):
            lr1=self.lr1
            # α=self.α
            γ=self.γ
            β=self.β
            δ=self.δ
            k=self.k
            # l=self.l
            n_max=self.n_max
            # n_min=self.n_min
        
            num_epochs=self.nep

            
            noise_mask = torch.zeros_like(self.score)
            # noise_mask[num_dims_to_optimize].requires_grad = False
           
            noise_mask.requires_grad_(True)
        
            optimizer = optim.Adam([
            {'params': [noise_mask], 'lr': lr1},
            
        ],betas=(0.9, 0.999))

            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                # noise_mask_pos=noise_mask.relu()
                noise_mask.retain_grad()
                l1=torch.norm(noise_mask,dim=0)
                
                mal_score=(self.score+noise_mask).requires_grad_(True)
                mal_score.retain_grad()
                
                # mal_rank=1e2*torch.sort(mal_score)[1].float().requires_grad_(True)
                # og_rank=1e2*torch.sort(self.score)[1].float().requires_grad_(True)
                # mal_rank.retain_grad()
                # og_rank.retain_grad()

                
                # l2=torch.norm(mal_rank-og_rank)
               
                l2=torch.norm(mal_score-k*torch.min(self.score))

                l3=torch.norm(mal_score-k*torch.max(self.score))
                l1=torch.norm(mal_score)
                l4=F.cosine_similarity(mal_score,self.score,dim=0)
                l5=torch.var(mal_score)

                # l1.retain_grad()
                l2.retain_grad()
                l3.retain_grad()
                l4.retain_grad()
                l5.retain_grad()
                

                loss=γ*(l2+l3)-δ*l4+β*l5 
                # loss=γ*(l2+l3)-δ*l4
                   
             
 
                loss.backward()
                optimizer.step()

                noise_mask.data=torch.clamp(noise_mask,min=-n_max,max=n_max)
                # mal_score.data = torch.clamp(mal_score.data, min(self.score), max(self.score))
                 
                
            # mal = mal.detach()
                # print(l1.item(),noise_mask)
            # noise_mask=noise_mask.detach()
            mal_score=mal_score.detach()       
            return mal_score
                

