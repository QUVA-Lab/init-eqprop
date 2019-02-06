from typing import Sequence

import numpy as np
import theano
import theano.tensor as tt

from artemis.experiments import ExperimentFunction, capture_created_experiments
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
from init_eqprop.result_display import report_score_from_result, \
    parse_eqprop_result, compare_learning_curves_new
from plato.core import symbolic


@symbolic
def energy(params, states, x, beta=0., y=None):
    energy_per_sample = sum(.5*(s_post**2).sum(axis=1) -tt.batched_dot(rho(s_pre).dot(w), rho(s_post)) - rho(s_post).dot(b) for s_pre, s_post, (w, b) in izip_equal([x] + states[:-1], states, params))
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
def equilibriating_step(params, states: Sequence[np.ndarray], x: np.ndarray, y=None, y_pressure=0.5, epsilon=0.5) -> Sequence[np.ndarray]:  # n_layers list of new layer states
    assert len(states)==len(params)
    state_grads = tt.grad(energy(params=params, states=states, x=x, beta=y_pressure, y=y).sum(), wrt=states)
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
def update_eq_params(x, params, negative_states, positive_states, learning_rates, beta):
    loss = (energy(params=params, states=positive_states, x=x) - energy(params=params, states=negative_states, x=x)).mean()
    return do_param_update(lambda p, pg, lr: p-lr*pg/beta, loss=loss, params = params, learning_rates=learning_rates)


def xavier_init(n_in, n_out, rng=None):
    rng = get_rng(rng)
    return rng.uniform(size=(n_in, n_out), low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)))


def initialize_params(layer_sizes, initial_weight_scale=1., rng=None):
    rng = get_rng(rng)
    params = [(initial_weight_scale*xavier_init(n_in, n_out, rng=rng), np.zeros(n_out)) for n_in, n_out in izip_equal(layer_sizes[:-1], layer_sizes[1:])]
    return params


@symbolic
def update_forward_params_with_energy(forward_params, eq_params, x, learning_rates, nonlinearity, disconnect_grads=True):  # Note that disconnect makes no difference - as the loss is calculated given the states.
    states = forward_pass(params=forward_params, x=x, disconnect_grads=disconnect_grads, nonlinearity=nonlinearity)
    loss = energy(params=eq_params, states=states, x=x).mean()
    return do_param_update(lambda p, pg, lr: p-lr*pg, loss=loss, params = forward_params, learning_rates=learning_rates)


@symbolic
def update_forward_params_with_contrast(forward_params, eq_states, x, nonlinearity, learning_rates, disconnect_grads=True):
    states = forward_pass(params=forward_params, x=x, nonlinearity=nonlinearity, disconnect_grads=disconnect_grads)
    loss = sum((.5*(fs-es)**2).sum(axis=1) for fs, es in izip_equal(states, eq_states)).mean(axis=0)
    return do_param_update(lambda p, pg, lr: p-lr*pg, loss=loss, params = forward_params, learning_rates=learning_rates)


def initialize_states(n_samples, noninput_layer_sizes):
    return [np.zeros((n_samples, dim)) for dim in noninput_layer_sizes]


@ExperimentFunction(compare=compare_learning_curves_new, one_liner_function=report_score_from_result, is_root=True, result_parser = parse_eqprop_result)
def demo_energy_based_initialize_eq_prop(
        n_epochs = 25,
        hidden_sizes = (500, ),
        minibatch_size = 20,
        beta = 0.5,
        epsilon = 0.5,
        learning_rate = (0.1, .05),
        n_negative_steps = 20,
        n_positive_steps = 4,
        initial_weight_scale = 1.,
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

    f_negative_eq_step = equilibriating_step.compile()
    f_inference_eq_step = equilibriating_step.compile()
    f_positive_eq_step = equilibriating_step.compile()
    f_parameter_update = update_eq_params.compile()
    f_forward_pass = forward_pass.partial(nonlinearity=forward_nonlinearity).compile()
    f_forward_parameter_update = update_forward_params_with_energy.partial(disconnect_grads=local_loss, nonlinearity=forward_nonlinearity).compile()
    f_forward_parameter_contrast_update = update_forward_params_with_contrast.partial(disconnect_grads=local_loss, nonlinearity=forward_nonlinearity).compile()
    f_energy = energy.compile()

    def do_inference(forward_params_, eq_params_, x, n_steps):
        states_ = forward_states_ = f_forward_pass(x=x, params=forward_params_) if train_with_forward else initialize_states(n_samples=x.shape[0], noninput_layer_sizes=layer_sizes[1:])
        for _ in range(n_steps):
            states_ = f_inference_eq_step(params=eq_params_, states = states_, x=x, epsilon=epsilon)
        return forward_states_[-1], states_[-1]

    results = Duck()
    for i, (ixs, info) in enumerate(minibatch_index_info_generator(n_samples=x_train.shape[0], minibatch_size=minibatch_size, n_epochs=n_epochs)):
        epoch = i*minibatch_size/x_train.shape[0]

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
            # if i % 200 == 0:
            #     with hold_dbplots():
            #         dbplot_collection(states, 'states', cornertext='NEG')
            #         dbplot(f_energy(params = eq_params, states=states, x=x_data_sample).mean(), 'energies', plot_type=DBPlotTypes.LINE_HISTORY_RESAMPLED)
            states = f_negative_eq_step(params=eq_params, states = states, x=x_data_sample, epsilon=epsilon)
        negative_states = states
        this_beta = rng.choice([-beta, beta]) if random_flip_beta else beta
        for t in range(n_positive_steps):
            # if i % 200 == 0:
            #     with hold_dbplots():
            #         dbplot_collection(states, 'states', cornertext='')
            #         dbplot(f_energy(params = eq_params, states=states, x=x_data_sample).mean(), 'energies', plot_type=DBPlotTypes.LINE_HISTORY_RESAMPLED)
            states = f_positive_eq_step(params=eq_params, states = states, x=x_data_sample, y=y_data_sample, y_pressure=this_beta, epsilon=epsilon)
        positive_states = states
        eq_params = f_parameter_update(x=x_data_sample, params=eq_params, negative_states=negative_states, positive_states=positive_states, learning_rates=learning_rate, beta=this_beta)

        # with hold_dbplots(draw_every=50):
        #     dbplot_collection([forward_params[0][0][:, :16].T.reshape(-1, 28, 28)] + [w for w, b in forward_params[1:]], '$\phi$')
        #     dbplot_collection([eq_params[0][0][:, :16].T.reshape(-1, 28, 28)] + [w for w, b in eq_params[1:]], '$\\theta$')
        #     dbplot_collection(forward_states, 'forward states')
        #     dbplot_collection(negative_states, 'negative_states')
        #     dbplot(np.array([f_energy(params = eq_params, states=forward_states, x=x_data_sample).mean(), f_energy(params = eq_params, states=negative_states, x=x_data_sample).mean()]), 'energies', plot_type=DBPlotTypes.LINE_HISTORY_RESAMPLED)

        if train_with_forward == 'contrast':
            forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=negative_states, learning_rates=learning_rate)
            # forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=negative_states, learning_rates=[lr/10 for lr in learning_rate])
        elif train_with_forward == 'contrast+':
            forward_params = f_forward_parameter_contrast_update(x=x_data_sample, forward_params=forward_params, eq_states=positive_states, learning_rates=learning_rate)
        elif train_with_forward == 'energy':
            forward_params = f_forward_parameter_update(x=x_data_sample, forward_params = forward_params, eq_params=eq_params, learning_rates=learning_rate)
        else:
            assert train_with_forward is False


# ======================================================================================================================

baseline_losstype = demo_energy_based_initialize_eq_prop.add_root_variant('losstype_baseline')
for random_flip_beta in (False, True):
    for forward_nonlinearity in ('rho(x)', 'rho(x)+0.01*x', 'sigm(x)'):
        for train_with_forward in (False, 'contrast', 'contrast+', 'energy'):
            baseline_losstype.add_variant(random_flip_beta=random_flip_beta, forward_nonlinearity=forward_nonlinearity, train_with_forward=train_with_forward)
                # X.add_variant(train_with_forward = False)
            # X=demo_energy_based_initialize_eq_prop.add_root_variant(forward_leak=leak)
            # X.add_variant(train_with_forward = 'energy')
            # X.add_variant(train_with_forward = 'contrast')
"""
Things learned:
- The rho(s)^2 term we were erroneously using earlier doesn't really matter.
- "energy" method fails always
- "contrast" seems to slightly help negative convergence
- "random_flip_beta" can actually help but can cause instability in the initialization network training (however forward_leak or sigmoid can fix this).
- "leaky rho" seems to be the most reliable nonlinearity, rho it seems CAN be unstable, sigm isn't quite as good.
- contrast+ behaves pretty similarily to contrast.  
"""

# ======================================================================================================================

with capture_created_experiments() as exps_reduced_steps:
    X = demo_energy_based_initialize_eq_prop.add_root_variant('stepsize_baseline', random_flip_beta=True)
    for n_negative_steps in (10, 5):
        X.add_variant(train_with_forward=False, n_negative_steps=n_negative_steps)
        XX=X.add_root_variant(train_with_forward='contrast', n_negative_steps=n_negative_steps)
        for forward_nonlinearity in ('rho(x)', 'rho(x)+0.01*x', 'sigm(x)'):
            XX.add_variant(forward_nonlinearity = forward_nonlinearity)


"""
Things learned:
- With few steps the normal rho(x) forward nonlinearity can really mess up.  Leaky-rho seems to be best.  
- Our init-network lets us train with fewer steps, and comes up with almost-as-good predicitons.  
- Very small degredation for 5-step case.
- leaky-rho gets best performance but may be less stable than sigmoid based off learning curve.   sigm is go
"""

# ======================================================================================================================

with capture_created_experiments() as deeper_experiments:
    X = demo_energy_based_initialize_eq_prop.add_root_variant('large',
        hidden_sizes = [500, 500, 500],
        n_epochs = 500,
        minibatch_size = 20,
        n_negative_steps = 500,
        n_positive_steps = 8,
        epsilon= .5,
        beta = 1.,
        learning_rate = [.128, .032, .008, .002]
        )

    for train_with_forward in (False, 'contrast'):
        for n_negative_steps in (10, 20, 100, 500):
            X.add_variant(train_with_forward=train_with_forward, n_negative_steps=n_negative_steps, forward_nonlinearity='rho(x)+0.01*x')
            X.add_variant(train_with_forward=train_with_forward, n_negative_steps=n_negative_steps, forward_nonlinearity='sigm(x)')

"""
What we learned
- Initialization works great!  Forward pass scores similarily to negative-tphase in the end
- Initialization allows learning with low numbers of negative steps.
- sigmoid initialization may work slightly better than leaky-rho, but not a big diff.
- There are limits... the 10-step version always fails.
"""

# ======================================================================================================================


with capture_created_experiments() as local_loss_exp:
    X = demo_energy_based_initialize_eq_prop.add_root_variant('local_loss_baseline', train_with_forward='contrast', random_flip_beta=False, forward_nonlinearity='rho(x)+0.01*x')
    X.add_variant(local_loss = False)
    X.add_variant(local_loss = True)
    X = demo_energy_based_initialize_eq_prop.add_root_variant('local_loss_baseline_energy', train_with_forward='energy', random_flip_beta=False, forward_nonlinearity='rho(x)+0.01*x')
    X.add_variant(local_loss = False)
    X.add_variant(local_loss = True)

"""
Does using only local loss hurt?
- Local loss works almost identically to global loss.  
- Local loss is not the reason the Energy approach is failing.
"""

# ======================================================================================================================

with capture_created_experiments() as exps_minimal_neg_convergence:
    X = demo_energy_based_initialize_eq_prop.add_root_variant('min_neg_baseline', train_with_forward='contrast+', forward_nonlinearity='rho(x)+0.01*x')
    for random_flip_beta in (False, True):
        for n_negative_steps in (10, 5, 2, 0):
            X.add_variant(random_flip_beta = random_flip_beta, n_negative_steps=n_negative_steps)
            X.add_variant(random_flip_beta = random_flip_beta, n_negative_steps=n_negative_steps, n_positive_steps = 10)
        # X.add_variant(n_negative_steps=5)
        # X.add_variant(n_negative_steps=0)
"""
What we learned:
- random_flip_beta brings down the required number of negative steps for stability (no surprise there)
- Using only 0 or 2 negative steps causes failure in every case.
"""


# ======================================================================================================================


baseline_pos_or_neg = demo_energy_based_initialize_eq_prop.add_root_variant('pos_or_neg', random_flip_beta=True, forward_nonlinearity='rho(x)+0.01*x')
for n_negative_steps in (0, 2, 4, 5, 10):
    for train_with_forward in ('contrast', 'contrast+'):
        baseline_pos_or_neg.add_variant(n_negative_steps=n_negative_steps, train_with_forward=train_with_forward)

"""
Is it better to predict the positive or negative phase state?
- Negative phase, thouugh they both work.
"""



# ======================================================================================================================

"""
It is surprising that the energy-based approach seems to fail so hard.  Why might this be?


It might be that the parameter Energy-gradient 
at the state given by the forward pass does not cause the states to move towards the minimizing direction...?
"""


if __name__ == '__main__':

    baseline_losstype.browse(display_format='args')
    # demo_energy_based_initialize_eq_prop.browse(raise_display_errors = True, truncate_result_to=150, display_format='args')
    # browse_experiments(exps_objective_comparison, truncate_result_to=80, raise_display_errors = True, display_format = 'args')
    # browse_experiments(exps_reduced_steps, display_format='args', )
    # browse_experiments(local_loss_exp, display_format='args', )
    # browse_experiments(deeper_experiments, display_format='args', truncate_result_to=80)
    # browse_experiments(local_loss_exp, display_format='args')
    # browse_experiments(exps_minimal_neg_convergence, display_format='args')

    # baseline_pos_or_neg.browse(display_format='args')

    # demo_energy_based_initialize_eq_prop.call(n_negative_steps=4, train_with_forward='contrast', random_flip_beta=True, forward_nonlinearity='rho(x)+0.01*x')
    # demo_energy_based_initialize_eq_prop.call(n_negative_steps=20, epsilon=0.1, train_with_forward='contrast', forward_nonlinearity='rho(x)+0.01*x')
    # demo_energy_based_initialize_eq_prop.call(n_negative_steps=10, epsilon=0.1, train_with_forward='contrast', forward_nonlinearity='rho(x)+0.01*x')
    # demo_energy_based_initialize_eq_prop.call(n_negative_steps=20, train_with_forward='energy', forward_nonlinearity='rho(x)+0.01*x')
