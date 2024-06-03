import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
 
 

class MalRankOptimizerMid:
    def __init__(self, rank,p,device):
        self.device = device
        self.rank = rank.float().requires_grad_(True).to(device)
        # self.lr=lr
        # self.k_s=k_s
        # self.n_s=n_s
        self.p=p


    def reverse_tensor(self, tensor):
        reverse_tensor=torch.flip(tensor, [0])
        reverse_tensor.retain_grad()
        return reverse_tensor.to(self.device)

    def cosine_similarity(self, tensor1, tensor2, dim=0):
        return F.cosine_similarity(tensor1, tensor2, dim=dim)
    
    def euclidean_distance(self, tensor1, tensor2, dim=0):
        squared_diff = (tensor1 - tensor2)**2
        sum_squared_diff = torch.sum(squared_diff, dim=dim)
        euclidean_dist = torch.sqrt(sum_squared_diff)
        return euclidean_dist

    
    def convert_to_binary(self,final):
        # Convert final to binary values
        binary_mask = torch.where(final < 0.5, torch.tensor(0).to(self.device), torch.tensor(1).to(self.device))
        return binary_mask
    
    def optimize_distance(self, num_epochs=100, lr=0.001,k=10):
       

        p = torch.nn.Parameter(torch.rand(1)).to(self.device)
        p.retain_grad()
        n=len(self.rank)

        mask_tensor = torch.bernoulli(torch.ones_like(self.rank) * p.item())
                                
        sorted_indices = torch.argsort(self.rank)
        sorted_indices=sorted_indices.float().requires_grad_(True).to(self.device)

        none_selected_indices = sorted_indices[:10]
        none_selected_indices = torch.cat((none_selected_indices, sorted_indices[-10:])).long()  # Appending last three indices

        selected_indiced=sorted_indices[int(n/2-k):int(n/2+k)].long()

        # Create a mask tensor with ones everywhere except at selected indices
        # mask_tensor = torch.ones_like(self.rank)
        mask_tensor[selected_indiced]=1
        mask_tensor[none_selected_indices] = 0

        mask_tensor_grad = torch.ones_like(mask_tensor, dtype=torch.bool)
        mask_tensor_grad[none_selected_indices] = False
        mask_tensor_grad[selected_indiced] = False
        mask_tensor.requires_grad_(True)

        # optimizer = optim.SGD([mask_tensor], lr=lr)
        optimizer = optim.Adam([{'params': [mask_tensor], 'lr': lr},{'params': [p], 'lr': lr}])
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            masked_input = self.rank * mask_tensor*torch.sigmoid(p)

            reversed=self.reverse_tensor(self.rank)

            masked_reverse=self.reverse_tensor(masked_input)

            euclidean_dist = torch.sum(self.euclidean_distance(reversed, self.rank))
            euclidean_dist_mid = torch.sum(self.euclidean_distance(masked_reverse, masked_input))
            euclidean_dist.retain_grad()
            euclidean_dist_mid.retain_grad()

            loss = euclidean_dist - euclidean_dist_mid 

            loss.backward()
            optimizer.step()
            
        final_mask = mask_tensor.detach()
        p=p.detach()
        return final_mask,p
   
   
    def optimize_p(self, num_epochs=100):
        # lr=self.lr
        n_s=40 
        k_s=200
        # α=self.α

        # p = torch.nn.Parameter(torch.rand(1)).to(self.device)
        # p.retain_grad()
        p=0.5
        n=len(self.rank)
        

        mask_tensor = torch.bernoulli(torch.ones_like(self.rank) * p)
                                
        # sorted_indices = torch.argsort(self.rank)
        # sorted_indices=sorted_indices.float().requires_grad_(True).to(self.device)

        none_selected_indices = mask_tensor[:n_s]
        none_selected_indices = torch.cat((none_selected_indices, mask_tensor[-n_s:])).long()  # Appending last three indices

        selected_indiced=mask_tensor[int(n/2-k_s):int(n/2+k_s)].long()

        # Create a mask tensor with ones everywhere except at selected indices
        # mask_tensor = torch.ones_like(self.rank)
        mask_tensor[selected_indiced]=1
        mask_tensor[none_selected_indices] = 0

        mask_tensor_grad = torch.ones_like(mask_tensor, dtype=torch.bool)
        mask_tensor_grad[none_selected_indices] = False
        mask_tensor_grad[selected_indiced] = False
        mask_tensor.requires_grad_(True)

        # optimizer = optim.SGD([mask_tensor], lr=lr)
        optimizer = optim.Adam([{'params': [mask_tensor], 'lr': 0.01}])
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # mask_tensor = torch.bernoulli(torch.ones_like(self.rank) * p.item())
            mask_tensor.requires_grad_(True)
            mask_tensor.retain_grad()           
            
            masked_input = self.rank * mask_tensor
            
            masked_reverse=self.reverse_tensor(masked_input)

            reversed=self.reverse_tensor(self.rank)
           
            cos_sim = torch.sum(self.cosine_similarity(reversed, self.rank))
            cos_sim_mid=torch.sum(self.cosine_similarity(masked_reverse, masked_input))
            cos_sim.retain_grad()
            cos_sim_mid.retain_grad()
            # regularization_term = torch.abs(torch.mean(torch.nonzero(mask_tensor)) - len(self.rank) / 2)

            # loss = -α*cos_sim+(1-α)*cos_sim_mid 
            loss=cos_sim_mid

            loss.backward()
            optimizer.step()
            
        final_mask = mask_tensor.detach()
        # final_mask.to(self.device)
        # p=p.detach()
        return final_mask,p
   
    def optimize_p_sorted(self, num_epochs=100):
        p=self.p

        # p = torch.nn.Parameter(torch.rand(1)).to(self.device)
        # p.retain_grad()
        # n=len(self.rank)

        sorted_rank=(torch.sort(self.rank)[1]+1).float().requires_grad_(True)

        mask_tensor = torch.bernoulli(torch.ones_like(self.rank) * p)
        mask_tensor.requires_grad_(True)
        mask_tensor.retain_grad()
                                

        optimizer = optim.Adam([{'params': [mask_tensor], 'lr': 0.01}])
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            
            masked_input = sorted_rank * mask_tensor
            
            # reversed=self.reverse_tensor(sorted_rank)

            masked_reverse=self.reverse_tensor(masked_input)

            # ########get the mal########
            #  # optimize_p(self, num_epochs=100, lr=0.001,k=10,α):

            # rounded_mask_tensor = self.convert_to_binary(mask_tensor).float()
            # masked_input = sorted_rank * rounded_mask_tensor
            # masked_input.requires_grad_(True)
            # masked_input.retain_grad()

            # # Flip all 1s to 0s and 0s to 1s to gettthe rest
            # flipped_mask_tensor = 1 - rounded_mask_tensor
            # flipped_mask_tensor.requires_grad_(True)
            # flipped_mask_tensor.retain_grad()

            # rest=sorted_rank*flipped_mask_tensor
            # rest.requires_grad_(True)
            # rest.retain_grad()  
            # mal=rest+masked_reverse
            # mal.requires_grad_(True)
            # mal.retain_grad()
                    

            ###########################

            # cos_sim = F.cosine_similarity(mal, sorted_rank, dim=0)
            cos_sim_mid= F.cosine_similarity(masked_reverse, masked_input, dim=0)
            
            # cos_sim.retain_grad()
            cos_sim_mid.retain_grad()
            # regularization_term = torch.abs(torch.mean(torch.nonzero(mask_tensor)) - len(self.rank) / 2)

            # loss = -α*cos_sim+(1-α)*cos_sim_mid 
            loss=cos_sim_mid

            loss.backward()
            optimizer.step()
            
        final_mask = mask_tensor.detach()
        # p=p.detach()
        return final_mask,p
   
