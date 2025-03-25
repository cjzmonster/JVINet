import torch
import torch.nn.functional as F
import torch.distributions as dist

action_logits = torch.rand(5)
# print(action_logits.shape)
action_probs = F.softmax(action_logits, dim=-1)
print(action_probs)

cate = dist.Categorical(action_probs)
action = cate.sample()
print(cate)
print(action)
print(cate.log_prob(action), torch.log(action_probs[action]))
