import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.noise_flag = False
        self.noise_std = 1.0
        
        # 用来加入参数噪声parameter noise
        if(self.noise_flag):
            self.noise_fc1 = torch.randn(self.fc1.weight.size())*self.noise_std
            self.noise_fc2 = torch.randn(self.fc2.weight.size())*self.noise_std
            self.noise_fc3 = torch.randn(self.fc3.weight.size())*self.noise_std
        
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def add_noise(self,weights, noise):                                                                                                                                                                                                                                              
        with torch.no_grad():                                                                                                                                                                                                                                                   
            weights.add_(noise)
    
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        """
        这里添加了parameter noise, 参考论文 
        Plappert M, Houthooft R, Dhariwal P, et al. Parameter space noise for exploration[J]. arXiv preprint arXiv:1706.01905, 2017.
        在这一部分每一层参数都会加入noise，每一层的输出都会进行normalization标准化
        """
        # 用来加入参数噪声parameter noise
        # torch.randn() Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
        # https://pytorch.org/docs/stable/generated/torch.randn.html
        
        if(self.noise_flag):
            self.noise_fc1 = torch.randn(self.fc1.weight.size())*self.noise_std
            self.noise_fc2 = torch.randn(self.fc2.weight.size())*self.noise_std
            self.noise_fc3 = torch.randn(self.fc3.weight.size())*self.noise_std
        
        if(self.noise_flag):
            self.add_noise(self.fc1.weight, self.noise_fc1)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        if(self.noise_flag):
            norm1 = nn.LayerNorm(h1.size())
            h1 = norm1(h1)
        
        if(self.noise_flag):
            self.add_noise(self.fc2.weight, self.noise_fc2)
        h2 = self.nonlin(self.fc2(h1))
        if(self.noise_flag):
            norm2 = nn.LayerNorm(h2.size())
            h2 = norm2(h2)
        
        out = self.out_fn(self.fc3(h2))
        return out