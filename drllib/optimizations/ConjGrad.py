from torch.optim.optimizer import Optimizer
import torch
import copy
import numpy as np
torch.manual_seed(0)
class ConjGrad(Optimizer):

    def __init__(self, params, lr=0.0001):
        defaults = dict(lr=lr)
        super(ConjGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ConjGrad, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if 'gradient_buffer' not in param_state:
                    buf = param_state['gradient_buffer'] = torch.clone(d_p).detach()
                    z = -buf
                    param_state['pbuf'] = torch.clone(z).detach()
                    param_state['step'] = 0

                buf = param_state['gradient_buffer']
                z = param_state['pbuf']
                param_state['step'] +=1

                if (d_p.dim() == 2):
                    Nt = d_p.shape[0] * d_p.shape[1]
                else:
                    Nt =  d_p.shape[0]

                if(param_state['step'] == 1):
                    p.data.add_(group['lr'], -d_p)

                elif (torch.div(param_state['step'], Nt)) == 0:
                    p.data.add_(group['lr'], -d_p)


                else:
                    if(d_p.dim() == 2):
                        x = d_p.view(d_p.shape[0] * d_p.shape[1],1)
                        bufx = buf.view(buf.shape[0] * buf.shape[1],1)
                    else:
                        x = d_p.view(d_p.shape[0], 1)
                        bufx = buf.view(buf.shape[0], 1)

                    nom = torch.mm(torch.t(x-bufx),x) + 0.001
                    denom = torch.mm(torch.t(bufx),bufx) + 0.001

                    beta = torch.div(nom,denom)


                    if(d_p.dim() == 1):

                        k = torch.squeeze(z,0)
                        bbeta = torch.squeeze(beta,0)

                        k.mul_(bbeta).add_(-d_p)
                        p.data.add_(group['lr'], k)
                    else:
                        z.mul_(beta).add_(-d_p)
                        zbuf = z.view(d_p.shape[0] * d_p.shape[1],1)
                        alpha = -torch.div(torch.mm(torch.t(zbuf), bufx), torch.mm(torch.t(zbuf), zbuf))
                        kalpha = torch.squeeze(alpha,0)
                        #print(kalpha.data)
                        p.data.add_(torch.mul(kalpha,z))
                param_state['gradient_buffer'] = torch.clone(d_p).detach()

        return loss
