import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
 
 

class MalRankOptimizerMid:
    def __init__(self, rank,device=torch.device('cuda')):
        self.device = device
        self.rank = rank.float().requires_grad_(True).to(device)


    def reverse_tensor(self, tensor):
        reverse_tensor=torch.flip(tensor, [0])
        reverse_tensor.retain_grad()
        return reverse_tensor.to(self.device)

    # def extract_reverse_combine(self, p):
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

        
        return combined_tensor.to(self.device)

    def cosine_similarity(self, tensor1, tensor2, dim=0):
        return F.cosine_similarity(tensor1, tensor2, dim=dim)
    
    def convert_to_binary(self,final):
        # Convert final to binary values
        binary_mask = torch.where(final < 0.5, torch.tensor(0).to(self.device), torch.tensor(1).to(self.device))
        return binary_mask
    
    def optimize_p(self, num_epochs=100, lr=0.001,k=10):
       

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

            cos_sim = torch.sum(self.cosine_similarity(reversed, self.rank))
            cos_sim_mid=torch.sum(self.cosine_similarity(masked_reverse, masked_input))
            cos_sim.retain_grad()
            cos_sim_mid.retain_grad()
            # regularization_term = torch.abs(torch.mean(torch.nonzero(mask_tensor)) - len(self.rank) / 2)

            loss = -cos_sim+cos_sim_mid 

            loss.backward()
            optimizer.step()
            
        final_mask = mask_tensor.detach()
        p=p.detach()
        return final_mask,p
   
   


