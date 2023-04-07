import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.utils import str_to_class
from deepinv.optim.data_fidelity import L2
from collections.abc import Iterable

class BaseOptim(nn.Module):
    r'''
        Class for optimisation algorithms that iterates the iterator.

        iterator : ...

        :param deepinv.optim.iterator iterator: description

    '''
    def __init__(self, iterator, max_iter=50, crit_conv='residual', thres_conv=1e-5, early_stop=True, F_fn = None, 
                anderson_acceleration=False, anderson_beta=1., anderson_history_size=5, verbose=False, return_dual=False,
                stepsize = 1., g_param=None, backtracking=False, gamma_backtracking = 0.1, eta_backtracking = 0.9,
                stepsize_g = None, params_dict={'stepsize': None, 'sigma': None}):

        super(BaseOptim, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.anderson_acceleration = anderson_acceleration
        self.F_fn = F_fn
        self.return_dual = return_dual
        self.iterator = iterator

        if isinstance(stepsize, Iterable):
            assert len(stepsize) >= max_iter, "stepsize must have at least max_iter elements"
            self.stepsize_iterable = True
            backtracking = False
        else :
            self.stepsize_iterable = False

        if isinstance(g_param, Iterable):
            assert len(g_param) >= max_iter, "g_param must have at least max_iter elements"
            self.g_param_iterable = True
        else :
            self.g_param_iterable = False

        # self.stepsize = torch.tensor(stepsize)
        # self.g_param = torch.tensor(g_param) if g_param else None
        # self.stepsize_g = torch.tensor(stepsize_g) if stepsize_g else None

        self.params_dict = {key: torch.tensor(value) if value is not None else None
                            for key, value in zip(params_dict.keys, params_dict.values)}

        # Now we have self.params_dict['stepsize']

        # def update_params_fn(cur_params, it, X, X_prev):
        #     if backtracking:
        #         x_prev, x = X_prev['est'][0], X['est'][0]
        #         F_prev, F = X_prev['cost'], X['cost']
        #         diff_F, diff_x = F_prev - F, (torch.norm(x - x_prev, p=2) ** 2).item()
        #         stepsize = cur_params['stepsize']
        #         if diff_F < (gamma_backtracking / stepsize) * diff_x :
        #             cur_params['stepsize'] = eta_backtracking * stepsize
        #     if self.stepsize_iterable:
        #         cur_params['stepsize'] = self.stepsize[it]
        #     if self.g_param_iterable:
        #         cur_params['g_param'] = self.g_param[it]
        #     return cur_params

        def update_params_fn(params_dict, it, X, X_prev):

            cur_params = self.get_params_it(params_dict, it)

            if backtracking:
                x_prev, x = X_prev['est'][0], X['est'][0]
                F_prev, F = X_prev['cost'], X['cost']
                diff_F, diff_x = F_prev - F, (torch.norm(x - x_prev, p=2) ** 2).item()
                stepsize = cur_params['stepsize']
                if diff_F < (gamma_backtracking / stepsize) * diff_x :
                    cur_params['stepsize'] = eta_backtracking * stepsize

            return cur_params



        if self.anderson_acceleration :
            self.anderson_beta = anderson_beta
            self.anderson_history_size = anderson_history_size
            self.fixed_point = AndersonAcceleration(self.iterator, update_params_fn=update_params_fn, max_iter=self.max_iter, history_size=anderson_history_size, beta=anderson_beta,
                            early_stop=early_stop, crit_conv=crit_conv, thres_conv=thres_conv, verbose=verbose)
        else :
            self.fixed_point = FixedPoint(self.iterator, update_params_fn=update_params_fn, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv, thres_conv=thres_conv, verbose=verbose)

    def get_params_it(self, params_dict, it):
        cur_params_dict = {key: value[it] if isinstance(value, nn.ModuleList) else value
                       for key, value in zip(params_dict.keys, params_dict.values)}
        return cur_params_dict

    def get_init(self, cur_params, y, physics):
        r'''
        '''
        x_init = physics.A_adjoint(y)
        init_X = {'est': (x_init,y), 'cost': self.F_fn(x_init,cur_params,y,physics) if self.F_fn else None} 
        return init_X

    def get_primal_variable(self, X):
        return X['est'][0]

    def get_dual_variable(self, X):
        return X['est'][1]

    def forward(self, y, physics, **kwargs):
        init_params = self.get_params_it(self.params_dict, 0)
        x = self.get_init(init_params, y, physics)
        x = self.fixed_point(x, self.params_dict, y, physics, **kwargs)
        return self.get_primal_variable(x) if not self.return_dual else self.get_dual_variable(x)

    def has_converged(self):
        return self.fixed_point.has_converged

def Optim(algo_name, data_fidelity=L2(), lamb=1., device='cpu', g=None, prox_g=None,
            grad_g=None, g_first=False, stepsize=1., g_param=None, stepsize_inter=1.,
            max_iter_inter=50, tol_inter=1e-3, beta=1., backtracking=False, gamma_backtracking = 0.1, 
            eta_backtracking = 0.9, F_fn=None, **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                grad_g=grad_g, g_first=g_first, stepsize_inter=stepsize_inter,
                max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta, F_fn = F_fn)

    optimizer = BaseOptim(iterator, F_fn = F_fn,  g_param=g_param,stepsize=stepsize, backtracking=backtracking,
                    gamma_backtracking = gamma_backtracking, eta_backtracking = eta_backtracking, **kwargs)

    return optimizer