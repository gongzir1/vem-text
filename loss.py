import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
 
 

class MalRankOptimizerMid:
    def __init__(self, rank):
        self.rank = rank
        self.rank = self.normalize(self.rank).float().requires_grad_(True)
        
    
    def normalize(self, tensor):
        normalized_tensor = tensor.clone().float()
        return normalized_tensor
    
    def get_mid(self, p):
        n = len(self.rank)
        mid_start = torch.round(n/2 - n * p).long()
        mid_end = torch.round(n/2 + p * n).long()
        mid_tensor = self.rank[mid_start:mid_end]
        mid_tensor.retain_grad()
        return mid_tensor


    def reverse_tensor(self, tensor):
        reverse_tensor=torch.flip(tensor, [0])
        reverse_tensor.retain_grad()
        return reverse_tensor

    def extract_reverse_combine(self, p):
        """
        Extract the mid tensor, reverse it, and combine it with the original rest tensor.

        Args:
        - p: Percentage of the tensor to extract as mid tensor.

        Returns:
        - Combined tensor.
        """
        n = len(self.rank)
        
        # Calculate start and end indices using straight-through estimator
        start_float = torch.round(n / 2 - n * p).long()
        end_float = torch.round(n / 2 + p * n).long()

        
        # # Use STE to approximate gradients through the rounding operation
        # start_int = start_float + torch.round(start_float).detach() - start_float.detach()
        # end_int = end_float + torch.round(end_float).detach() - end_float.detach(
        

        
        mid_tensor = self.rank[start_float:end_float]
        reversed_mid_tensor = torch.flip(mid_tensor, [0])
        rest_tensor_first = self.rank[:start_float]
        rest_tensor_last = self.rank[end_float:]

        combined_tensor = torch.cat((rest_tensor_first, reversed_mid_tensor, rest_tensor_last), dim=0)
        mid_tensor.retain_grad()
        reversed_mid_tensor.retain_grad()
        combined_tensor.retain_grad()

        
        return combined_tensor

    def cosine_similarity(self, tensor1, tensor2, dim=0):
        return F.cosine_similarity(tensor1, tensor2, dim=dim)
    
    def convert_to_binary(self,final):
        # Convert final to binary values
        binary_mask = torch.where(final < 0.5, torch.tensor(0), torch.tensor(1))
        return binary_mask
    
    def optimize_p(self, num_epochs=100, lr=0.01):
       
        # Initial mask tensor (random initialization)
        # mask_tensor = torch.rand(len(self.rank), requires_grad=True)
        # p = torch.nn.Parameter(torch.tensor(0.5))
        p = torch.nn.Parameter(torch.rand(1))
        p.retain_grad()
        mask_tensor = torch.bernoulli(torch.ones_like(self.rank) * p.item())
                                
        mask_tensor.requires_grad_(True)
        mask_tensor.retain_grad()

       
        # optimizer = optim.SGD([mask_tensor], lr=lr)
        optimizer = optim.Adam([{'params': [mask_tensor], 'lr': lr},{'params': [p], 'lr': lr}])
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            masked_input = self.rank * mask_tensor*torch.sigmoid(p)

            # masked_input = self.rank * mask_tensor
            
            reversed=self.reverse_tensor(self.rank)

            masked_reverse=self.reverse_tensor(masked_input)

            cos_sim = torch.sum(self.cosine_similarity(reversed, self.rank))
            cos_sim_mid=torch.sum(self.cosine_similarity(masked_reverse, masked_input))
            cos_sim.retain_grad()
            cos_sim_mid.retain_grad()

            loss = -cos_sim+cos_sim_mid + torch.sum((binary_mask[:-1] - binary_mask[1:]).abs())

    
            # loss=-self.cosine_similarity(self.rank,masked_input)

            loss.backward()
            optimizer.step()
            
        final_mask = mask_tensor.detach()
        p=p.detach()
        return final_mask,p
   
   

# # Example usage 
rank = torch.tensor([1,3,2,5,4,6,7,8,9,10,11,23,12,13])  # Example rank tensor
optimizer = MalRankOptimizerMid(rank)

final,p = optimizer.optimize_p()

rounded_mask_tensor = optimizer.convert_to_binary(final)


# Flip all 1s to 0s and 0s to 1s
flipped_mask_tensor = 1 - rounded_mask_tensor
rest=rank*flipped_mask_tensor
print(rest)
print("Optimized mask:", rounded_mask_tensor)
print(p)











