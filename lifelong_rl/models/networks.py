import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init

from lifelong_rl.torch import pytorch_util as ptu
from lifelong_rl.torch.modules import LayerNorm


def identity(x):
    return x

class BatchEnsembleLinear(nn.Module):

    def __init__(self, input_size, output_size, ensemble_size, bias=True):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.ensemble_size = ensemble_size

        self.W = nn.Parameter(torch.empty(output_size, input_size))  # m*n
        self.r = nn.Parameter(torch.empty(ensemble_size, input_size))  # M*m
        self.s = nn.Parameter(torch.empty(ensemble_size, output_size))  # M*n

        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, X):
        """
        Expects input in shape (B*M, C_in), dim 0 layout:
            ------ x0, model 0 ------
            -------x0, model 1 ------
                      ...
            ------ x1, model 0 ------
            -------x1, model 1 ------
                      ...
        """
        B = X.shape[0] // self.ensemble_size
        X = X.view(B, self.ensemble_size, -1)  # Reshape input to (B, M, C_in)
        R = self.r.unsqueeze(0)  # Add a dimension for broadcasting
        S = self.s.unsqueeze(0)  # Add a dimension for broadcasting
        bias = self.bias.unsqueeze(0)  # Add a dimension for broadcasting

        # Eq. 5 from BatchEnsembles paper
        output = torch.matmul((X * R), self.W.t()) * S + bias  # (B, M, C_out)
        
        # Flatten output back to (B*M, C_out)
        output = output.view(B * self.ensemble_size, -1)
        return output

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W,gain=nn.init.calculate_gain('relu'))
        # Another way to initialize the fast weights
        #nn.init.normal_(self.r, mean=1., std=0.1)
        #nn.init.normal_(self.s, mean=1., std=0.1)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        if True:
            with torch.no_grad():
              # random sign initialization from paper
                self.r.bernoulli_(0.5).mul_(2).add_(-1)
                self.s.bernoulli_(0.5).mul_(2).add_(-1)
        else:
            # nn.init.normal_(self.r, mean=1., std=0.5)
            # nn.init.normal_(self.s, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)


class BatchEnsembleLinearPlus(nn.Module):

    def __init__(self, input_size, output_size, ensemble_size, bias=True, diversity = False):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.ensemble_size = ensemble_size
        self.weight_diversity = diversity

        self.W = nn.Parameter(torch.empty(output_size, input_size))  # m*n
        self.r = nn.Parameter(torch.empty(ensemble_size, input_size))  # M*m
        self.s = nn.Parameter(torch.empty(ensemble_size, output_size))  # M*n

        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, X):
        """
        Expects input in shape (B*M, C_in), dim 0 layout:
            ------ x0, model 0 ------
            -------x0, model 1 ------
                      ...
            ------ x1, model 0 ------
            -------x1, model 1 ------
                      ...
        """
        B = X.shape[0] // self.ensemble_size
        X = X.view(B, self.ensemble_size, -1)  # Reshape input to (B, M, C_in)
        R = self.r.unsqueeze(0)  # Add a dimension for broadcasting
        S = self.s.unsqueeze(0)  # Add a dimension for broadcasting
        bias = self.bias.unsqueeze(0)  # Add a dimension for broadcasting

        # Eq. 5 from BatchEnsembles paper
        output = torch.matmul((X * R), self.W.t()) * S + bias  # (B, M, C_out)
        
        # Flatten output back to (B*M, C_out)
        output = output.view(B * self.ensemble_size, -1)
        diver =  torch.tensor(0) 
        if self.weight_diversity:
          R1 = self.r/torch.norm(self.r,dim=1,keepdim=True)
          S1 = self.s/torch.norm(self.s,dim=1,keepdim=True)
          diver = 1 - (torch.mean(torch.matmul(R1,R1.t()) + torch.matmul(S1,S1.t())))/2

        return output,diver

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W,gain=nn.init.calculate_gain('relu'))
        # Another way to initialize the fast weights
        #nn.init.normal_(self.r, mean=1., std=0.1)
        #nn.init.normal_(self.s, mean=1., std=0.1)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        if True:
            with torch.no_grad():
              # random sign initialization from paper
                self.r.bernoulli_(0.5).mul_(2).add_(-1)
                self.s.bernoulli_(0.5).mul_(2).add_(-1)
        else:
            # nn.init.normal_(self.r, mean=1., std=0.5)
            # nn.init.normal_(self.s, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)


class BatchEnsembleFlattenMLPPlus(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            layer_norm=None,
            batch_norm=False,
            norm_input=False,
            obs_norm_mean=None,
            obs_norm_std=None,
            hidden_activate=F.gelu, # THANH
            diversity_regularize = False # THANH
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.ensemble_num = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        self.sampler = np.random.default_rng()

        self.hidden_activation = hidden_activate
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.norm_input = norm_input
        if self.norm_input:
            self.obs_norm_mean, self.obs_norm_std = ptu.from_numpy(obs_norm_mean), ptu.from_numpy(obs_norm_std + 1e-6)

        self.fcs = []
        self.diversity_regularize= diversity_regularize

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = BatchEnsembleLinearPlus(
                ensemble_size=ensemble_size,
                input_size=in_size,
                output_size=next_size,
                diversity = self.diversity_regularize,
            )
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = BatchEnsembleLinearPlus(
            ensemble_size=ensemble_size,
            input_size=in_size,
            output_size=output_size,
            diversity = self.diversity_regularize,
        )


    def forward(self, *inputs, **kwargs):
        if self.norm_input:
            obs = (inputs[0] - self.obs_norm_mean) / self.obs_norm_std
            flat_inputs = torch.cat([obs, inputs[1]], dim=-1)
        else:
            flat_inputs = torch.cat([inputs[0], inputs[1]], dim=-1)
        
        if kwargs.get("sample", False):
            flat_inputs = flat_inputs.repeat_interleave(self.ensemble_size, 0)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        diversity = 0
        for _, fc in enumerate(self.fcs):
            h,div = fc(h)
            diversity +=div 
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation,div = self.last_fc(h)
        diversity +=div
        output = self.output_activation(preactivation)
        return output,diversity

    def sample(self, *inputs):
        preds,*_ = self.forward(*inputs,sample=True)
        B = preds.shape[0] // self.ensemble_size
        #(B*Self.ensemble_size,1) => (self.ensemble_size, B, 1)
        preds = preds.view(B, self.ensemble_size, -1)
        # Return min, mean and std of the ensemble
        return torch.min(preds, dim=1)[0],0#,preds.mean(dim=1), preds.std(dim=1)
    
    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError

class BatchEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
            norm_input=False,
            obs_norm_mean=None,
            obs_norm_std=None,
            hidden_activate=torch.relu, 
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        self.sampler = np.random.default_rng()

        self.hidden_activation = hidden_activate
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.norm_input = norm_input
        if self.norm_input:
            self.obs_norm_mean, self.obs_norm_std = ptu.from_numpy(obs_norm_mean), ptu.from_numpy(obs_norm_std + 1e-6)

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = BatchEnsembleLinear(
                ensemble_size=ensemble_size,
                input_size=in_size,
                output_size=next_size,
            )
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = BatchEnsembleLinear(
            ensemble_size=ensemble_size,
            input_size=in_size,
            output_size=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *inputs, **kwargs):
        """Calculate the forward pass of Q(s, a).

        Args:
            inputs: list[observation,action]: list of tensors containing the observation and action size B x obs_dim , B x act_dim 

        Returns:
            Q(s,a): return Q(s,a) size B x 1, where emsamble members output are stack along dim 0 [q_m0 , q_m1, ...,q_mN, q_m0, q_m1, ...]^T 
        """
        
        if self.norm_input:
            obs = (inputs[0] - self.obs_norm_mean) / self.obs_norm_std
            flat_inputs = torch.cat([obs, inputs[1]], dim=-1)
        else:
            flat_inputs = torch.cat([inputs[0], inputs[1]], dim=-1)

        dim=len(flat_inputs.shape)
        if kwargs.get("sample",False):
            flat_inputs = flat_inputs.repeat_interleave(self.ensemble_size,0)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs,sample=True)
        B = preds.shape[0] // self.ensemble_size
        #(B*Self.ensemble_size,1) => (self.ensemble_size, B, 1)
        preds = preds.view(B,self.ensemble_size,-1 )
        # Return min, mean and std of the ensemble
        return torch.min(preds, dim=1)[0],preds.mean(dim=1), preds.std(dim=1)
 


    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError


class MimoEnsembleFlattenMLP(nn.Module):
    def __init__(self,
                 ensemble_size,
                 hidden_sizes,
                 input_size,
                 output_size,
                 layer_norm=None,
                 batch_norm=False,
                 final_init_scale=None,
                 norm_input=False,
                 obs_norm_mean=None,
                 obs_norm_std=None,
                 width_multiplier=1):
        super(MimoEnsembleFlattenMLP, self).__init__()
        self.ensemble_num = ensemble_size
        self.hidden_activation = torch.tanh

        self.input_layer = nn.Linear(
            input_size*ensemble_size, hidden_sizes[0]*width_multiplier)
        self.backbone_model = BackboneModel(
            [layer_size*width_multiplier for layer_size in hidden_sizes], hidden_activation=self.hidden_activation)
        self.norm_input = norm_input
        if self.norm_input:
            self.obs_norm_mean, self.obs_norm_std = ptu.from_numpy(
                obs_norm_mean), ptu.from_numpy(obs_norm_std + 1e-6)
        self.output_layer = nn.Linear(
            hidden_sizes[-1]*width_multiplier, output_size * ensemble_size)
        self.output_activation = identity
        # initialize weights
        init.xavier_uniform_(self.input_layer.weight)
        self.input_layer.bias.data.fill_(0)
        init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0)

    def forward(self, *inputs, **kwargs):
        """Calculate the forward pass of Q(s, a).

        Inputs will be grouped by ensemble member, and then passed through the network. Each input will have a corresponding output. The network shares the same body 
        for all inputs, but each input has its own head.

        Args:
            inputs: list[observation,action]: list of tensors containing the observation and action size B x obs_dim , B x act_dim 

        Returns:
            Q(s,a): return Q(s,a) size B x 1, where emsamble members output are stack along dim 0 [q_m0(x0) , q_m1(x1), ...,q_mN(xM), q_m0(xM+1), q_m1, ...]^T 
        """
        inputs = [inputs[0], inputs[1]]
        if self.norm_input:
            inputs[0] = (inputs[0] - self.obs_norm_mean) / self.obs_norm_std

        inputs = torch.cat(inputs, dim=-1)

        if kwargs.get("sample", None) == True:
            inputs = inputs.repeat_interleave(self.ensemble_num, 0)

        return self.forward_(inputs)

    def forward_(self, input):
        dim = len(input.shape)
        # transform B*E to B//M*(E*M)
        B, E, *_ = input.shape
        M = self.ensemble_num
        h = input.view(B//M, -1)

        # standard feedforward network
        h = self.input_layer(h)
        h = self.hidden_activation(h)
        h = self.backbone_model(h)
        h = self.output_layer(h)
        output = self.output_activation(h)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)
        return output.view(B, -1)

    def sample(self, *inputs):
        preds = self.forward(*inputs, sample=True)
        B = preds.shape[0] // self.ensemble_num
        return torch.min(preds.view(B, self.ensemble_num, -1), dim=1)


class BackboneModel(nn.Module):
    def __init__(self, hidden_dim, hidden_activation=F.relu):
        super(BackboneModel, self).__init__()
        self.hidden_activation = hidden_activation
        for i, (in_dim, out_dim) in enumerate(zip(hidden_dim[:-1], hidden_dim[1:])):
            self.add_module(f"l{i}", nn.Linear(in_dim, out_dim))
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, layer in self.named_children():
            x = layer(x)
            x = self.hidden_activation(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.fcs = []
        self.batch_norms = []

        # data normalization
        self.input_mu = nn.Parameter(
            ptu.zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            ptu.ones(input_size), requires_grad=False).float()

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight, w_scale)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.batch_norm:
                bn = nn.BatchNorm1d(next_size)
                self.__setattr__('batch_norm%d' % i, bn)
                self.batch_norms.append(bn)

            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            ptu.orthogonal_init(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = (input - self.input_mu) / (self.input_std + 1e-6)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std != std] = 0
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std = mask * std + (1-mask) * np.ones(self.input_size)
        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class Ensemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList()
        for m in models:
            self.models.append(m)

        self.ensemble_size = len(models)
        self.input_size = self.models[0].input_size
        self.output_size = self.models[0].output_size

    def forward(self, input):
        preds = ptu.zeros(
            (len(self.models), *input.shape[:-1], self.output_size))
        for i in range(len(self.models)):
            preds[i] = self.models[i].forward(input)
        return preds

    def sample(self, input):
        preds = self.forward(input)
        inds = torch.randint(0, len(self.models), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])
        samples = (inds == 0).float() * preds[0]
        for i in range(1, len(self.models)):
            samples += (inds == i).float() * preds[i]
        return samples

    def fit_input_stats(self, data, mask=None):
        for m in self.models:
            m.fit_input_stats(data, mask=mask)


class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # data normalization
        self.input_mu = nn.Parameter(
            ptu.zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            ptu.ones(input_size), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            layer_size = (ensemble_size, in_size, next_size)
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                w_std_value=1/(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            w_std_value=1/(2*np.sqrt(in_size)),
            b_init_value=b_init_value,
        )

    def forward(self, input):
        dim = len(input.shape)

        # input normalization
        h = (input - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, output_size)
        return output

    def sample(self, input):
        preds = self.forward(input)

        inds = torch.randint(0, len(self.elites), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])

        samples = (inds == 0).float() * preds[self.elites[0]]
        for i in range(1, len(self.elites)):
            samples += (inds == i).float() * preds[self.elites[i]]

        return samples

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)


class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
            norm_input=False,
            obs_norm_mean=None,
            obs_norm_std=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity

        self.layer_norm = layer_norm

        self.norm_input = norm_input
        if self.norm_input:
            self.obs_norm_mean, self.obs_norm_std = ptu.from_numpy(
                obs_norm_mean), ptu.from_numpy(obs_norm_std + 1e-6)

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                ptu.orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        inputs = [inputs[0], inputs[1]]
        if self.norm_input:
            inputs[0] = (inputs[0] - self.obs_norm_mean) / self.obs_norm_std

        flat_inputs = torch.cat(inputs, dim=-1)
        state_dim = inputs[0].shape[-1]

        dim = len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)
        return output

    def sample(self, *inputs):
        preds = self.forward(*inputs)
        return torch.min(preds, dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError



class BatchEnsembleLinearRank1(nn.Module):

    def __init__(self, input_size, output_size, ensemble_size, bias=True, diversity = False):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.ensemble_size = ensemble_size
        self.weight_diversity = diversity

        self.W = nn.Parameter(torch.empty(output_size, input_size))  # m*n
        self.r = nn.Parameter(torch.empty(ensemble_size, input_size))  # M*m
        self.s = nn.Parameter(torch.empty(ensemble_size, output_size))  # M*n

        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, X):
        '''
        X: (B, M, C_in)
        return (B, M, C_out)

        '''
        R = self.r.unsqueeze(0)  # Add a dimension for broadcasting
        S = self.s.unsqueeze(0)  # Add a dimension for broadcasting
        bias = self.bias.unsqueeze(0)  # Add a dimension for broadcasting

        # Eq. 5 from BatchEnsembles paper
        output = torch.matmul((X * R), self.W.t()) * S + bias  # (B, M, C_out)

        diver =  torch.tensor(0) 
        if self.weight_diversity:
          R1 = self.r/torch.norm(self.r,dim=1,keepdim=True)
          S1 = self.s/torch.norm(self.s,dim=1,keepdim=True)
          diver = 1 - (torch.mean(torch.matmul(R1,R1.t()) + torch.matmul(S1,S1.t())))/2

        return output,diver

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.W,gain=nn.init.calculate_gain('relu'))
        # Another way to initialize the fast weights
        #nn.init.normal_(self.r, mean=1., std=0.1)
        #nn.init.normal_(self.s, mean=1., std=0.1)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        if True:
            with torch.no_grad():
              # random sign initialization from paper
                self.r.bernoulli_(0.5).mul_(2).add_(-1)
                self.s.bernoulli_(0.5).mul_(2).add_(-1)
        else:
            # nn.init.normal_(self.r, mean=1., std=0.5)
            # nn.init.normal_(self.s, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)
            nn.init.normal_(self.r, mean=1., std=0.5)


class BatchEnsembleFlattenRank1(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            layer_norm=None,
            batch_norm=False,
            norm_input=False,
            obs_norm_mean=None,
            obs_norm_std=None,
            hidden_activate=F.gelu, # THANH
            diversity_regularize = False # THANH
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.ensemble_num = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        self.sampler = np.random.default_rng()

        self.hidden_activation = hidden_activate
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.norm_input = norm_input
        if self.norm_input:
            self.obs_norm_mean, self.obs_norm_std = ptu.from_numpy(obs_norm_mean), ptu.from_numpy(obs_norm_std + 1e-6)

        self.fcs = []
        self.diversity_regularize= diversity_regularize

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = BatchEnsembleLinearRank1(
                ensemble_size=ensemble_size,
                input_size=in_size,
                output_size=next_size,
                diversity = self.diversity_regularize,
            )
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = BatchEnsembleLinearRank1(
            ensemble_size=ensemble_size,
            input_size=in_size,
            output_size=output_size,
            diversity = self.diversity_regularize,
        )


    def forward(self, *inputs, **kwargs):
        if self.norm_input:
            obs = (inputs[0] - self.obs_norm_mean) / self.obs_norm_std
            flat_inputs = torch.cat([obs, inputs[1]], dim=-1)
        else:
            flat_inputs = torch.cat([inputs[0], inputs[1]], dim=-1)

        flat_inputs = flat_inputs.repeat_interleave(self.ensemble_size, 0).view(-1,self.ensemble_size, self.input_size)

        # input normalization
        h = flat_inputs

        # standard feedforward network
        diversity = 0
        for _, fc in enumerate(self.fcs):
            h,div = fc(h)
            diversity +=div 
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation,div = self.last_fc(h)
        diversity +=div
        output = self.output_activation(preactivation) # (B,M, C_out)
        # Transpose to (M, B, C_out)
        output = output.transpose(0, 1).contiguous()
        
        return output #,diversity

    def sample(self, *inputs):
        preds = self.forward(*inputs)
        return torch.min(preds, dim=0)[0]
    
    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError
