from abc import abstractmethod
from functools import partial

from torch.nn import functional as F

from artemis.general.nested_structures import NestedType, get_leaves_and_rebuilder, get_leaves
from artemis.general.should_be_builtins import izip_equal
import torch


def nested_grad(outputs, inputs, allow_unused = False, **kwargs):
    input_struct = NestedType.from_data(inputs)
    grads = torch.autograd.grad(outputs=outputs, inputs = input_struct.get_leaves(inputs), allow_unused=allow_unused, **kwargs)
    structured_grads = input_struct.expand_from_leaves(grads, check_types = not allow_unused)
    return structured_grads


class GradientBasedOptimizer(object):

    def __init__(self):
        self._states = {}

    def update(self, params, grads=None, loss=None, break_grads = True, inplace=False, states='internal'):
        """
        Update the parameters based on the gradients.
        :param params: Parameters to update
        :param grads: The gradients for each parameter
        :param Optional[Tensor] loss:  For convenience, you may specify loss instead of gradients.
        :param break_grads: Do not propagate gradient information through the updata.
        :param inplace:
        :return: new params: The updated parameters.

        """

        params, param_rebuilder = get_leaves_and_rebuilder(params)

        assert (loss is None) != (grads is None), "You must specify either loss or grads."
        if loss is not None:
            grads = torch.autograd.grad(loss, inputs=params)
        else:
            grads = get_leaves(grads)

        use_state_key = isinstance(states, str)

        if use_state_key:  # The optimizer saves states internally
            state_key = states
            states = [None]*len(params) if state_key not in self._states else self._states[state_key]
        else:
            states = [None]*len(params) if states is None else states

        with torch.autograd.set_grad_enabled(not break_grads):
            if inplace:
                for i, (p, g, state) in enumerate(izip_equal(params, grads, states)):
                    if g is not None:
                        new_param, states[i] = self._compute_param_update(p, g, state)
                        p.data = new_param.data
                if use_state_key:
                    self._states[state_key] = states
                    return
                else:
                    return states
            else:
                new_param_values, states = zip(*(self._compute_param_update(p, g, state) if g is not None else p for p, g, state in izip_equal(params, grads, states)))
                if break_grads:
                    for p in new_param_values:  # Now make sure child parameter do keep grads.
                        p.requires_grad = True
                new_param_values = param_rebuilder(new_param_values)
                if use_state_key:
                    self._states[state_key] = states
                    return new_param_values
                else:
                    return new_param_values, states

    def update_(self, params, grads=None, loss=None):
        params = list(params)
        self.update(params, grads=grads, loss=loss, inplace=True)

    @abstractmethod
    def _compute_param_update(self, param, grad, state):
        """"""


class GradientDescent(GradientBasedOptimizer):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        GradientBasedOptimizer.__init__(self)

    def _compute_param_update(self, param, grad, _):
        return param - self.learning_rate*grad, None


class BoundedGradientDescent(GradientBasedOptimizer):

    def __init__(self, learning_rate, bound):
        min_bound, max_bound = bound
        self.bound = (min_bound, max_bound)
        self.learning_rate = learning_rate
        GradientBasedOptimizer.__init__(self)

    def _compute_param_update(self, param, grad, _):
        return torch.clamp(param - self.learning_rate*grad, *self.bound), None


class MomentumOptimizer(GradientBasedOptimizer):

    def __init__(self, learning_rate, momentum, weight_decay=None, bound = None):
        assert bound is None or len(bound)==2 and bound[0]<bound[1]
        assert momentum is None or 0<=momentum<1
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.bound = bound
        GradientBasedOptimizer.__init__(self)

    def _compute_param_update(self, param, grad, momentum_term, break_grad=True):
        if momentum_term is None:
            momentum_term = torch.zeros_like(param)
        momentum_term = momentum_term * self.momentum + grad
        new_param = param - self.learning_rate*momentum_term
        if self.weight_decay is not None:
            new_param = new_param - self.weight_decay*new_param
        if self.bound is not None:
            new_param = torch.clamp(new_param, *self.bound)
        return new_param, momentum_term


class AdaGradOptimizer(GradientBasedOptimizer):

    def __init__(self, learning_rate, max_scaling=1e5, decay_rate = 0., weight_decay=None):
        self.eps = torch.tensor(1./max_scaling)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        # self.sum_squared_grad = None
        GradientBasedOptimizer.__init__(self)

    def _compute_param_update(self, param, grad, sum_squared_grad, break_grad=True):
        # sum_squared_grad, initialized = create_optimizer_param_like(param)
        if sum_squared_grad is None:
            sum_squared_grad = torch.zeros_like(param)
        new_sum_squared_grad = (1-self.decay_rate)*sum_squared_grad + grad**2
        scale = torch.max(self.eps, torch.sqrt(new_sum_squared_grad))
        new_param = param - (self.learning_rate / scale) * grad
        if self.weight_decay is not None:
            new_param = new_param - self.weight_decay*new_param
        return new_param, new_sum_squared_grad
        # return [(param, param - (self.learning_rate / scale) * grad), (sum_squared_grad, new_ssg)]


class RMSPropOptimizer(GradientBasedOptimizer):

    def __init__(self, learning_rate = 0.1, decay = 0.9, max_scaling = 1e5):
        super(RMSPropOptimizer, self).__init__()
        self.decay = decay
        self.epsilon = torch.tensor(1./max_scaling)
        self.learning_rate = learning_rate

    def _compute_param_update(self, param, grad, state):
        mean_squared_grad = torch.zeros_like(param) if state is None else state
        new_mean_squared_grad = self.decay*mean_squared_grad + (1-self.decay) * grad**2
        new_param = param - self.learning_rate*grad / torch.max(torch.sqrt(new_mean_squared_grad), self.epsilon)
        return new_param, new_mean_squared_grad


class AdamMaxOptimizer(GradientBasedOptimizer):

    def __init__(self, alpha = 1e-3, beta_1=0.1, beta_2=0.001, eps = 1e-8, weight_decay=None):
        super(AdamMaxOptimizer, self).__init__()
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._weight_decay = weight_decay

    def _compute_param_update(self, param, grad, state, break_grad=True):

        if state is None:
            mom1 = torch.zeros_like(param)
            mom2 = torch.zeros_like(param)
        else:
            mom1, mom2 = state

        if self._weight_decay is not None:
            grad = grad + self._weight_decay*.5*(param**2).sum()
        mom1_new = mom1 + self._beta_1*(grad-mom1)
        mom2_new = torch.max(torch.abs(grad) + self._eps, (1.-self._beta_2)*mom2)
        new_param = param - self._alpha * mom1_new / mom2_new

        return new_param, (mom1_new, mom2_new)


def string_to_function(func_str):

    return lambda x: eval(func_str, {'x': x}, {'rho': partial(torch.clamp, min=0, max=1), 'clip': partial(torch.clamp, min=0, max=1), 'relu': F.relu, 'sigm': torch.nn.functional.sigmoid, 'lin': (lambda x: x)})


def get_named_nonlinearity(name):
    return {
        'relu': F.relu,
        'clip': partial(torch.clamp, min=0, max=1),
        'lin': lambda x: x,
        'sigm': torch.nn.functional.sigmoid
    }[name]