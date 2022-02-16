from torch.optim.optimizer import Optimizer
import torch

class LM(Optimizer):

    def __init__(self, params, lr=0.01,input = None, model = None, nparam = None, beta = 0.5,target=None ):
        defaults = dict(lr=lr,input = input, model = model, beta = beta,target = target, nparam = nparam)
        super(LM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LM, self).__setstate__(state)

    def criter_loss(self,val, out):
        soft_val = torch.max(val)
        soft_error = (soft_val - out) ** 2
        return soft_error

    def get_jacobian(self,net, x):
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


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            model = group['model']
            lr = group['lr']
            inputs = group['input']
            target = group['target']
            total_inputs = inputs.shape[0]
            beta = group['beta']
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rloss = torch.zeros((total_inputs, 1), dtype=torch.float)
            J = torch.zeros((total_inputs,total_params), dtype=torch.float)
            for i, data in enumerate(inputs):
                datar = data.unsqueeze(0)
                J[i,:] = torch.FloatTensor(self.get_jacobian(model,datar))
                values = model(datar)
                rloss[i] = self.criter_loss(values, target[i])
            rloss = rloss.reshape(-1)
            rightprt = torch.matmul(-torch.t(J), rloss)
            leftpart = torch.inverse(torch.matmul(torch.t(J), J) + beta * torch.eye(total_params, total_params))
            change = torch.matmul(leftpart, rightprt)
            ind = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-lr * change[ind])
                ind += 1
        return torch.sum(rloss)/total_inputs