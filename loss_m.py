import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function

class MalOptimizer: 
    def __init__(self,score,device,α,γ,β,δ,lr1,lr2,k_s,v,scale1,scale2,nep,l_c):
        self.device = device
        self.score = score.requires_grad_(True).to(device)
        self.α=α
        self.γ=γ
        self.β=β             
        # self.p=p
        self.δ=δ
        self.lr1=lr1
        self.lr2=lr2
        self.v=v
        self.k_s=k_s
        self.scale1=scale1
        self.scale2=scale2
        self.nep=nep
        self.l_c=l_c
        
    

    def optimize_mask_on_m(self):
            lr1=self.lr1
            lr2=self.lr2
            α=self.α
            γ=self.γ
            β=self.β
            p=0.5
            δ=self.δ
            k_s=self.k_s
            self.nep=50
            num_epochs=self.nep

            v=self.v
            scale1=self.scale1
            scale2=self.scale2
            l_c=self.l_c

            # mask_tensor = torch.bernoulli(torch.ones_like(self.score) * p)
            mask_tensor = torch.zeros_like(self.score)

            mask_tensor.requires_grad_(True)
            mask_tensor.retain_grad()
            mal=self.score
            mask_tensor_b=mask_tensor
            loss='ed'
            d= torch.tensor(0.01, dtype=torch.float)
            d.requires_grad_(True)
            d.retain_grad()
                                  
            # optimizer = optim.Adam([{'params': [mask_tensor], 'lr': 0.01}])
            optimizer = optim.SGD([
            {'params': [mask_tensor], 'lr': lr1},
            {'params': [d], 'd': lr2},       
        ],betas=(0.9, 0.999))

            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                # mask_n=2*mask_tensor-1
                # mask_n.retain_grad()
                mean=torch.mean(mask_tensor)
                mean.retain_grad()
                mask_tensor_b=torch.relu(k_s*(mask_tensor))
                     
               
                mask_tensor_b.retain_grad()
                

                mean_b=torch.mean(mask_tensor_b)
                            
                masked_input = self.score * mask_tensor_b
                masked_input.retain_grad()
                mask_input_smallest=abs(torch.mean(masked_input)-torch.max(self.score))
                mask_input_largest=abs(torch.mean(masked_input)-torch.min(self.score))
                # mask_mean=torch.mean(torch.abs(masked_input))
                variance = torch.var(masked_input)
                mask_input_largest.retain_grad()
                mask_input_smallest.retain_grad()
                variance.retain_grad()

                # masked_mal=masked_input*math.log(d.item())
                # masked_mal=-masked_input/math.exp(d.item())
                masked_mal=-(masked_input/abs(d.item()))
                ed_d=torch.norm(masked_mal - masked_input)
                loss_constrain=torch.abs(ed_d - (torch.max(self.score)-torch.min(self.score)))

                masked_mal.retain_grad()
                mal=self.score+masked_mal
                mal.requires_grad_(True)
                mal.retain_grad()
                
                cos_sim = scale2*F.cosine_similarity(mal, self.score, dim=0)
                cos_sim.retain_grad()
                if loss=='cosine':
           
                    cos_masked=1000*F.cosine_similarity(masked_mal,masked_input,dim=0)
                    cos_masked.retain_grad()
                    
                    loss=-α*cos_sim+γ*cos_masked-β*(mask_input_smallest+mask_input_largest)+v*variance+l_c*loss_constrain
                    # loss=-α*cos_sim-γ*diff_mean
                    print(mask_tensor_b)
                else:
                    euclidean_loss = 1000*torch.norm(masked_mal - masked_input)
                    euclidean_loss.retain_grad()
                    
                    loss=-α*cos_sim-γ*euclidean_loss-β*(mask_input_smallest+mask_input_largest)+v*variance+l_c*loss_constrain
                    # loss=-α*cos_sim-γ*diff_mean
                    # print(mask_tensor_b)
                     
 
                loss.backward()
                optimizer.step()
                 # Apply a transformation to enforce range constraints
                mask_tensor.data = torch.clamp(mask_tensor.data, 0, 1)
                
            # mal = mal.detach()
            mask_tensor_b=mask_tensor_b.detach()
            d=d.detach()
            
            
            return mask_tensor_b,d
                

