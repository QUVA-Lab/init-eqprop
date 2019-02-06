from typing import Sequence
import numpy as np
import theano
import theano.tensor as tt
from matplotlib import pyplot as plt
from artemis.experiments import ExperimentFunction, capture_created_experiments
from artemis.experiments.experiment_record_view import make_record_comparison_duck
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.duck import Duck
from artemis.general.nested_structures import NestedType
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal
from artemis.general.speedometer import Speedometer
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.tools.costs import percent_argmax_incorrect
from artemis.ml.tools.iteration import minibatch_index_info_generator
from artemis.plotting.expanding_subplots import subplot_at, set_figure_border_size
from init_eqprop.result_display import report_score_from_result, \
    compare_learning_curves_new, parse_eqprop_result
from plato.core import symbolic


@symbolic
def energy(params, states, fwd_states, x, forward_deviation_cost, zero_deviation_cost, beta=0., y=None):
    energy_per_sample = sum(.5*zero_deviation_cost*(s_post**2).sum(axis=1) + .5*forward_deviation_cost*((s_post-s_fwd)**2).sum(axis=1) -tt.batched_dot(rho(s_pre).dot(w), rho(s_post)) - rho(s_post).dot(b) for s_pre, s_post, s_fwd, (w, b) in izip_equal([x] + states[:-1], states, fwd_states, params))
    if y is not None:
        energy_per_sample = energy_per_sample + beta * ((states[-1] - y) ** 2).sum(axis=1)
    return energy_per_sample


@symbolic
def rho(x):
    return tt.clip(x, 0, 1)


@symbolic
def forward_pass(params, x, nonlinearity, disconnect_grads = True):
    nonlinearity_func = lambda x: eval(nonlinearity, dict(np=np, rho=rho, sigm=tt.nnet.sigmoid), dict(x=x))
    if disconnect_grads:
        return [s for s in [x] for w, b in params for s in [nonlinearity_func(theano.gradient.disconnected_grad(s).dot(w)+b)]]
    else:
        return [s for s in [x] for w, b in params for s in [nonlinearity_func(s.dot(w)+b)]]


@symbolic
def equilibriating_step(params, states: Sequence[np.ndarray], x: np.ndarray, fwd_states, forward_deviation_cost, zero_deviation_cost, y=None, y_pressure=0.5, epsilon=0.5) -> Sequence[np.ndarray]:  # n_layers list of new layer states
    assert len(states)==len(params)
    state_grads = tt.grad(energy(params=params, states=states, fwd_states=fwd_states, forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost, x=x, beta=y_pressure, y=y).sum(), wrt=states)
    new_states = [tt.clip(s - epsilon*g, 0, 1) for s, g in izip_equal(states, state_grads)]
    return new_states


@symbolic
def do_param_update(update_function, loss, params, learning_rates):
    """
    :param update_function: Has form new_param = update_function(param, param_grad, learning_rate)
    :param loss: A scalar loss
    :param params: A structure of parameters
    :param learning_rates: A learning rate or structure of learning rates which can be broadcast against params
    :return: A structure of new parameter values in the same format as params.
    """
    param_structure = NestedType.from_data(params)
    flat_params = param_structure.get_leaves(params)
    flat_param_grads = tt.grad(loss, wrt=flat_params)
    new_flat_params = (update_function(p, pg, lr) for p, pg, lr in izip_equal(flat_params, flat_param_grads, param_structure.get_leaves(learning_rates, broadcast=True)))
    new_params = param_structure.expand_from_leaves(new_flat_params)
    return new_params


@symbolic
def update_eq_params(x, params, negative_states, positive_states, fwd_states, learning_rates, beta, forward_deviation_cost, zero_deviation_cost):
    loss = (energy(params=params, states=positive_states, x=x, fwd_states=fwd_states, forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost) - energy(params=params, states=negative_states, x=x, fwd_states=fwd_states, forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost)).mean()
    return do_param_update(lambda p, pg, lr: p-lr*pg/beta, loss=loss, params = params, learning_rates=learning_rates)


def xavier_init(n_in, n_out, rng=None):
    rng = get_rng(rng)
    return rng.uniform(size=(n_in, n_out), low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)))


def initialize_params(layer_sizes, initial_weight_scale=1., rng=None):
    rng = get_rng(rng)
    params = [(initial_weight_scale*xavier_init(n_in, n_out, rng=rng), np.zeros(n_out)) for n_in, n_out in izip_equal(layer_sizes[:-1], layer_sizes[1:])]
    return params


# @symbolic
# def update_forward_params_with_energy(forward_params, eq_params, x, learning_rates, disconnect_grads=True):  # Note that disconnect makes no difference - as the loss is calculated given the states.
#     states = forward_pass(params=forward_params, x=x, disconnect_grads=disconnect_grads)
#     loss = energy(params=eq_params, states=states, x=x).mean()
#     return do_param_update(lambda p, pg, lr: p-lr*pg, loss=loss, params = forward_params, learning_rates=learning_rates)


@symbolic
def update_forward_params_with_contrast(forward_params, eq_states, x, nonlinearity, learning_rates, disconnect_grads=True):
    states = forward_pass(params=forward_params, x=x, nonlinearity=nonlinearity, disconnect_grads=disconnect_grads)
    loss = sum((.5*(fs-es)**2).sum(axis=1) for fs, es in izip_equal(states, eq_states)).mean(axis=0)
    return do_param_update(lambda p, pg, lr: p-lr*pg, loss=loss, params = forward_params, learning_rates=learning_rates)


def initialize_states(n_samples, noninput_layer_sizes):
    return [np.zeros((n_samples, dim)) for dim in noninput_layer_sizes]


@ExperimentFunction(compare=compare_learning_curves_new, one_liner_function=report_score_from_result, is_root=True, result_parser = parse_eqprop_result)
def demo_energy_based_initialize_eq_prop_fwd_energy(
        n_epochs = 25,
        hidden_sizes = (500, ),
        minibatch_size = 20,
        beta = 0.5,
        epsilon = 0.5,
        learning_rate = (0.1, .05),
        n_negative_steps = 20,
        n_positive_steps = 4,
        initial_weight_scale = 1.,
        forward_deviation_cost = 0.,
        zero_deviation_cost = 1,
        epoch_checkpoint_period = {0: .25, 1: .5, 5: 1, 10: 2, 50: 4},
        n_test_samples = 10000,
        skip_zero_epoch_test = False,
        train_with_forward = 'contrast',
        forward_nonlinearity = 'rho(x)',
        local_loss = True,
        random_flip_beta = True,
        seed = 1234,):

    print('Params:\n' + '\n'.join(list(f'  {k} = {v}' for k, v in locals().items())))

    assert train_with_forward in ('contrast', 'contrast+', 'energy', False)

    rng = get_rng(seed)
    n_in = 784
    n_out = 10

    dataset = get_mnist_dataset(flat=True, n_test_samples=None).to_onehot()
    x_train, y_train = dataset.training_set.xy
    x_test, y_test = dataset.test_set.xy  # Their 'validation set' is our 'test set'

    if is_test_mode():
        x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]
        n_epochs=1

    layer_sizes = [n_in] + list(hidden_sizes) + [n_out]

    eq_params = initialize_params(layer_sizes=layer_sizes, initial_weight_scale=initial_weight_scale, rng=rng)
    forward_params = initialize_params(layer_sizes=layer_sizes, initial_weight_scale=initial_weight_scale, rng=rng)

    y_train = y_train.astype(np.float32)

    sp = Speedometer(mode='last')
    is_epoch_checkpoint = Checkpoints(epoch_checkpoint_period, skip_first=skip_zero_epoch_test)

    f_negative_eq_step = equilibriating_step.partial(forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost).compile()
    f_inference_eq_step = equilibriating_step.partial(forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost).compile()
    f_positive_eq_step = equilibriating_step.partial(forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost).compile()
    f_parameter_update = update_eq_params.partial(forward_deviation_cost=forward_deviation_cost, zero_deviation_cost=zero_deviation_cost).compile()
    f_forward_pass = forward_pass.partial(nonlinearity=forward_nonlinearity).compile()
    # f_forward_parameter_update = update_forward_params_with_energy.compile()
    f_forward_parameter_contrast_update = update_forward_params_with_contrast.partial(nonlinearity=forward_nonlinearity).compile()

    def do_inference(forward_params_, eq_params_, x, n_steps):
        states_ = forward_states_ = f_forward_pass(x=x, params=forward_params_) if train_with_forward else initialize_states(n_samples=x.shape[0], noninput_layer_sizes=layer_sizes[1:])
        for _ in range(n_steps):
            states_ = f_inference_eq_step(params=eq_params_, states = states_, fwd_states = forward_states_, x=x, epsilon=epsilon)
        return forward_states_[-1], states_[-1]

    results = Duck()
    # last_time, last_epoch = time(), -1
    for i, (ixs, info) in enumerate(minibatch_index_info_generator(n_samples=x_train.shape[0], minibatch_size=minibatch_size, n_epochs=n_epochs)):
        epoch = i*minibatch_size/x_train.shape[0]

        # print(f'Training Rate: {(time()-last_time)/(epoch-last_epoch):3g}s/ep')
        # last_time, last_epoch = time(), epoch

        if is_epoch_checkpoint(epoch):
            n_samples = n_test_samples if n_test_samples is not None else len(x_test)
            (test_init_error, test_neg_error), (train_init_error, train_neg_error) = [
                [percent_argmax_incorrect(prediction, y[:n_test_samples]) for prediction in do_inference(forward_params_=forward_params, eq_params_=eq_params, x=x[:n_test_samples], n_steps=n_negative_steps)]
                for x, y in [(x_test, y_test), (x_train, y_train)]
                ]
            print(f'Epoch: {epoch:.3g}, Iter: {i}, Test Init Error: {test_init_error:.3g}%, Test Neg Error: {test_neg_error:.3g}%, Train Init Error: {train_init_error:.3g}%, Train Neg Error: {train_neg_error:.3g}%, , Mean Rate: {sp(i):.3g}iter/s')
            results[next, :] = dict(iter=i, epoch=epoch, test_init_error=test_init_error, test_neg_error=test_neg_error, train_init_error=train_init_error, train_neg_error=train_neg_error)
            yield results
            if epoch>2 and train_neg_error>50:
                return

        # The Original training loop, just taken out here:
        x_data_sample, y_data_sample = x_train[ixs], y_train[ixs]

        states = forward_states = f_forward_pass(x=x_data_sample, params=forward_params) if train_with_forward else initialize_states(n_samples=minibatch_size, noninput_layer_sizes=layer_sizes[1:])
        for t in range(n_negative_steps):
            states = f_negative_eq_step(params=eq_params, states = states, x=x_data_sample, epsilon=epsilon, fwd_states=forward_states)
        negative_states = states
        this_beta = rng.choice([-beta, beta]) if random_flip_beta else beta
        for t in range(n_positive_steps):
            states = f_positive_eq_step(params=eq_params, states = states, x=x_data_sample, y=y_data_sample, y_pressure=this_beta, epsilon=epsilon, fwd_states=forward_states)
        positive_states = states
        eq_params = f_parameter_update(x=x_data_sample, params=eq_params, negative_states=negative_states, positive_states=positive_states, fwd_states=forward_states, learning_rates=learning_rate, beta=this_beta)

        if train_with_forward == 'contrast':
            forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=negative_states, learning_rates=learning_rate)
            # forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=negative_states, learning_rates=[lr/10 for lr in learning_rate])
        elif train_with_forward == 'contrast+':
            forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=positive_states, learning_rates=learning_rate)
        # elif train_with_forward == 'energy':
        #     forward_params = f_forward_parameter_update(x=x_data_sample, forward_params = forward_params, eq_params=eq_params, learning_rates=learning_rate)
        else:
            assert train_with_forward is False


# ======================================================================================================================

baseline_scale = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('scale_baseline', forward_nonlinearity = 'rho(x)+0.01*x')
for forward_deviation_cost in [0., 0.1, 1, 10, 100]:
    baseline_scale.add_variant(forward_deviation_cost = forward_deviation_cost)

"""
What we wanted to find out:
- How does adding a forward deviation cost affect performance?
- What is an appropriate range to set it at?

What we learned:
- 10, 100 cause learning to fail
- Seems there is not a significant difference for the others, so probably 1 is an appropriate value to try for remaining
  experiments since it's the largets cost that does not impede learning.  

"""

# ======================================================================================================================


baseline_nstep = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('step_baseline', forward_nonlinearity = 'rho(x)+0.01*x')
for forward_deviation_cost in (0, 1):
    for n_negative_steps in (10, 5, 2, 1):
        baseline_nstep.add_variant(forward_deviation_cost = forward_deviation_cost, n_negative_steps = n_negative_steps)

baseline_nstep.add_variant(forward_deviation_cost=3, n_negative_steps=1)

"""
What we aim to find out: 
- Does adding a forward_deviation cost allow us to reduce the number of negative steps, by allowing the forward network 
  to make better direct predictions of the equilibrium netowrk state?

What we learned:
- Great news!  The loss term allows stable learning with 2 negative steps.  
- Seems there's little or no penalty for decreasing number of steps given that you're using the cost.
- Possibly a mild score disadvantage to using forward_deviation_cost  
- There are limits... 1 negative step does not work.
"""

# ======================================================================================================================

baseline_just_around_forward = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('baseline_just_around_forward', forward_nonlinearity = 'rho(x)+0.01*x', forward_deviation_cost=1, n_negative_steps=10)
for zero_deviation_cost in (1, 0.1, 0):
    baseline_just_around_forward.add_variant(zero_deviation_cost=zero_deviation_cost)

"""
What we aim to find out: 
- Can we just remove the normal squared loss term and replace it with the "around forward pass" loss term?

What we learned:
- Nope, you need that zero-pull (training fails without)
- Setting it small (0.1) also causes failure....

"""
# ======================================================================================================================

baseline_zero_neg = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('baseline_zero_neg', forward_nonlinearity = 'rho(x)+0.01*x', train_with_forward='contrast+')
for forward_deviation_cost in (0, 1):
    for n_negative_steps in (0, 2):
        baseline_zero_neg.add_variant(n_negative_steps=n_negative_steps, forward_deviation_cost=forward_deviation_cost)
    baseline_zero_neg.add_variant(forward_deviation_cost=forward_deviation_cost, n_negative_steps = 0, n_positive_steps = 20)
"""
What we aim to find out: 
- Can we move to a "zero negative step" convergence scheme?

What we learned:
- No, this doesn't just work out of the box.  
- It does work with a small number of negative steps.
"""

# ======================================================================================================================

baseline_final = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('baseline_final', forward_nonlinearity = 'rho(x)+0.01*x')
baseline_final_small = baseline_final.add_root_variant('small', n_negative_steps=50)
baseline_final_large = baseline_final.add_root_variant('large',
        hidden_sizes = [500, 500, 500],
        n_epochs = 500,
        minibatch_size = 20,
        n_negative_steps = 500,
        n_positive_steps = 8,
        epsilon= .5,
        beta = 1.,
        learning_rate = [.128, .032, .008, .002]
        )


with capture_created_experiments() as exps:
    for train_with_forward in (False, 'contrast'):
        baseline_final_small.add_variant(train_with_forward=train_with_forward, n_negative_steps = 20)
        baseline_final_small.add_variant(train_with_forward=train_with_forward, n_negative_steps = 4)
        baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 500)
        baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 20)
        baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 10)


# ======================================================================================================================

# Run time for big net:
# 50neg, 8 pos: ~220s/epoch.  So 100 epochs would take ~6.1hrs, plus all the tests but those should amortize so maybe 8hrs?

baseline_final_final = demo_energy_based_initialize_eq_prop_fwd_energy.add_root_variant('baseline_final_final', forward_nonlinearity = 'rho(x)+0.01*x')
baseline_final_small = baseline_final_final.add_root_variant('small', n_epochs=50)
baseline_final_large = baseline_final_final.add_root_variant('large',
        hidden_sizes = [500, 500, 500],
        n_epochs = 200,
        minibatch_size = 20,
        n_negative_steps = 500,
        n_positive_steps = 8,
        epsilon= .5,
        beta = 1.,
        learning_rate = [.128, .032, .008, .002]
        )


with capture_created_experiments() as exps:
    for train_with_forward in (False, 'contrast'):

        # for forward_deviation_cost in np.arange(0, 2, 0.125) if train_with_forward=='contrast' else [0]:
        # for forward_deviation_cost in [0] + list(1.5**np.arange(10)/8) if train_with_forward=='contrast' else [0]:
        for forward_deviation_cost in (0, 0.1, 1) if train_with_forward=='contrast' else [0]:

        # for forward_deviation_cost in [0] + list(1.5**np.arange(10)/8) if train_with_forward=='contrast' else [0]:
            baseline_final_small.add_variant(train_with_forward=train_with_forward, n_negative_steps = 20, forward_deviation_cost=forward_deviation_cost)
            baseline_final_small.add_variant(train_with_forward=train_with_forward, n_negative_steps = 10, forward_deviation_cost=forward_deviation_cost)
            baseline_final_small.add_variant(train_with_forward=train_with_forward, n_negative_steps = 4, forward_deviation_cost=forward_deviation_cost)
            baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 50, forward_deviation_cost=forward_deviation_cost)
            baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 20, forward_deviation_cost=forward_deviation_cost)
            baseline_final_large.add_variant(train_with_forward=train_with_forward, n_negative_steps = 10, forward_deviation_cost=forward_deviation_cost)


# for e in exps:
#     if e.get_args()['train_with_forward']:
#         e.add_variant(forward_deviation_cost=0)
#         e.add_variant(forward_deviation_cost=0.1)


def compare_lambdas(records):

    duck = make_record_comparison_duck(records)

    plt.figure(figsize=(8, 5))

    set_figure_border_size(hspace=0.4, top=0.1, bottom=0.1, left=0.1)
    for row, netsize in enumerate(('small', 'large')):
        for col, n_negative_steps in enumerate(sorted(set(duck[duck[:, 'exp_id'].map(lambda x: netsize in x), 'args', 'n_negative_steps']))):
            data = []
            for forward_deviation_cost in sorted(set(duck[:, 'args', 'forward_deviation_cost'])):

                filter_ixs = \
                    (duck[:, 'args', 'n_negative_steps'].each_eq(n_negative_steps)) \
                    & (duck[:, 'args', 'forward_deviation_cost'].each_eq(forward_deviation_cost)) \
                    & (duck[:, 'args', 'train_with_forward'].each_eq('contrast')) \
                    & (duck[:, 'exp_id'].map(lambda x: netsize in x))

                if not any(filter_ixs):
                    continue

                this_result = duck[filter_ixs].only()['result']
                data.append((forward_deviation_cost, this_result[-1, 'test_init_error'], this_result[-1, 'test_neg_error']))

            if len(data)==0:
                continue
            fwd_cost, test_init_error, test_neg_error = zip(*data)

            subplot_at(row, col)
            plt.title(f'{netsize}: $T^-={n_negative_steps}$')
            plt.plot(fwd_cost, test_init_error, label='Init Eq. Prop: $s^f$')
            plt.plot(fwd_cost, test_neg_error, label='Init Eq. Prop: $s^-$')
            plt.ylim(0, 10)
            plt.grid()
            plt.xlabel('$\lambda$')
            plt.ylabel('Test Score')
            plt.legend()

    plt.show()


if __name__ == '__main__':
    # baseline_scale.browse(display_format='args', )
    # baseline_final.browse(display_format='args', )
    # baseline_final_small.browse()
    baseline_final_final.browse()
    # baseline_final_final.browse(display_format='args', )
    # baseline_nstep.browse(display_format='args', )
    # baseline_just_around_forward.browse(display_format='args', )
    # baseline_zero_neg.browse(display_format='args', )
    # baseline_final_final.browse()
    # baseline_final_small.browse()
    # records = [r for r in baseline_final_final.get_variant_records(only_last=True, only_completed=True, ).values() if r is not None]
    # compare_lambdas(records)
