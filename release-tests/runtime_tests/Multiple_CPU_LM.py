import torch
from torch.optim.optimizer import Optimizer
import multiprocessing as mp
import numpy as np

"""
E.g.
    optimizer=LM_mp(params=model.parameters(),model=model,input = x,target=y, nof_epochs=500,beta = 0.01,lr=0.01)
    MSE=optimizer.optimize_parallel()
"""
class LM_mp(Optimizer):

    def __init__(self, params, lr=0.01, input=None, model=None, nof_epochs=1, nparam=None, beta=0.5, target=None):

        self.nof_epochs = nof_epochs

        defaults = dict(lr=lr, input=input, model=model, beta=beta, target=target, nparam=nparam)
        super(LM_mp, self).__init__(params, defaults)

    current_g_index = 0
    d = []
    H = []
    total_cost = 0

    def __setstate__(self, state):
        super(LM_mp, self).__setstate__(state)

    def get_Gradient_R(self, net, x):
        """
        Compute gradient for every parameter of network stack them in order
        as a list

        Note, x is a single input.

        Parameters
        ----------
        net : pytorch nn
            Neural network object as a pytorch computation graph

        x : input tensor
            single input sample to the network.

        Returns
        -------
        cols : python list of gradient tensors
            row vector containing the gradient of the forward model 'net'
        """

        y = net.forward(x)
        y.backward()
        groups = net.parameters()
        cols = []
        for g in groups:
            flat_grad = g.grad.reshape(-1)
            for fl_gr in flat_grad:
                cols.append(fl_gr.data.clone())
        params = net.parameters()
        for p in params:
            p.grad.data.zero_()
        return cols

    def get_cumulative_d_H_loss(self, inputs, targets, total_params):
        """
        Parameters
        ----------
        inputs : numpy ndarray of inputs
            inputs to the network

        targets :numpy ndarray of target outputs

        total_params : int
            total number of trainable parameters

        Returns
        -------
        d_c: pytorch tensor of shape = (total_params,)
            Cumulative difference vector that corresponds to sum_per_input((f(x)-y)grad(f))

        H_c: pytorch tensor of shape = (total_params,total_params)
            Cumulative Pseudo_Hessian matrix that corresponds to sum_per_input(grad(f)*grad(f)')

        loss_c: pytorch tensor of cumulative cost
            Cumulative cost
        """

        """convert inputs and targets to tensors from numpy array"""
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        """get relevant parameters"""
        group = self.param_groups[self.current_g_index]
        model = group['model']
        total_inputs = inputs.shape[0]

        err = torch.zeros((total_inputs, 1), dtype=torch.float)  # net(x)-y
        J = torch.zeros((total_inputs, total_params), dtype=torch.float)

        for i, data in enumerate(inputs):
            """Get Gradients"""
            datar = data.unsqueeze(0)
            J[i, :] = torch.FloatTensor(self.get_Gradient_R(model, datar))

            """Calculate Error as specified"""
            net_out = model(datar)
            err[i] = (targets[i] - net_out)

        """Calculate cumulative d, and cumulative Hessian"""
        err = err.reshape(-1)

        d_c = torch.matmul(-torch.t(J), err).detach()
        H_c = torch.matmul(torch.t(J), J).detach()
        loss_c = torch.sum(err ** 2).detach()

        return (d_c, H_c, loss_c)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        """get relevant parameters"""
        group = self.param_groups[self.current_g_index]
        lr = group['lr']
        beta = group['beta']

        """calculate the change in step"""
        change = torch.matmul(torch.inverse(self.H + beta * torch.eye(self.H.shape[0])), self.d)
        parameter_index = 0

        for p in group['params']:
            if p.requires_grad != True:
                continue

            param_shape = tuple(p.shape)
            flat_params = p.data.reshape(-1)
            param_length = flat_params.shape[0]

            p.data.add_(-lr * change[parameter_index:parameter_index + param_length].reshape(param_shape))
            parameter_index += param_length
        return self.total_cost

    def optimize_parallel(self):

        nof_cpus = mp.cpu_count()

        with mp.Pool(nof_cpus) as p:

            for epoch in range(self.nof_epochs):

                for g_index in range(len(self.param_groups)):

                    self.current_g_index = g_index
                    group = self.param_groups[self.current_g_index]
                    model = group['model']
                    inputs = group['input']
                    targets = group['target']
                    total_inputs = inputs.shape[0]
                    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    """Shuffle and Split Data"""

                    shuffle_index = np.random.permutation(total_inputs)
                    inputs = inputs[shuffle_index, :]
                    targets = targets[shuffle_index, :]

                    data_partition = np.append(np.arange(0, total_inputs, total_inputs / nof_cpus, dtype='int32'),
                                               total_inputs)

                    mp_inputs = [(inputs[data_partition[i]:data_partition[i + 1], :], \
                                  targets[data_partition[i]:data_partition[i + 1], :], \
                                  total_params) for i in range(nof_cpus)]

                    """Calculate d and H"""
                    partial_d_H_loss = p.starmap(self.get_cumulative_d_H_loss, mp_inputs)

                    """Average d and H"""
                    self.d = torch.zeros((total_params,), dtype=torch.float)
                    self.H = torch.zeros((total_params, total_params), dtype=torch.float)
                    self.total_cost = 0

                    for i in range(nof_cpus):
                        self.d = self.d + partial_d_H_loss[i][0] / total_inputs
                        self.H = self.H + partial_d_H_loss[i][1] / total_inputs
                        self.total_cost = self.total_cost + partial_d_H_loss[i][2]

                    """Call parameter update with step method"""
                    MSE = self.step() / total_inputs
        return MSE
