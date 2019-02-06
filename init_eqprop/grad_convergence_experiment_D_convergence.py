import torch
from artemis.experiments import ExperimentFunction
from artemis.experiments.experiment_record import ExperimentRecord
from artemis.general.checkpoint_counter import do_every
from artemis.general.duck import Duck
from artemis.general.should_be_builtins import izip_equal
from artemis.plotting.db_plotting import dbplot, hold_dbplots, dbplot_collection
from artemis.plotting.expanding_subplots import vstack_plots, add_subplot, select_subplot, hstack_plots
from artemis.plotting.pyplot_plus import get_color_cycle_map, outside_right_legend
from init_eqprop.pytorch_helpers import MomentumOptimizer, AdamMaxOptimizer, AdaGradOptimizer
from matplotlib import pyplot as plt
import numpy as np

def cosine_distance(a, b, dim):
    d = (a*b).sum(dim=dim)/torch.sqrt((a**2).sum(dim=dim) * (b**2).sum(dim=dim))
    return d


def xavier_init(n_in, n_out):
    return torch.sqrt(torch.tensor(2./(n_in+n_out))) * torch.randn(n_in, n_out, requires_grad=True)


def ortho_init(n_in, n_out):
    return torch.nn.init.orthogonal_(torch.empty((n_in, n_out), requires_grad = True))


def param_init(n_in, n_out, mode, init_scale):
    return {'ortho': ortho_init, 'xavier': xavier_init}[mode](n_in, n_out)*init_scale


_nonlinearities = {'relu': torch.relu, 'tanh': torch.tanh, 'lin': (lambda x: x), 'leaky_clip': (lambda x: torch.clamp(x, 0, 1)+0.01*x), 'centered_leaky_clip': (lambda x: torch.clamp(x, -.5, .5)+0.01*x), 'leaky_relu': (lambda x: torch.relu(x)+0.01*x)}


def get_named_nonlinearity(name):
    return _nonlinearities[name]

def forward_pass(phi, x, nonlinearity):
    h = _nonlinearities[nonlinearity]
    return [s for s in [x] for w in phi for s in [h(s @ w)]]



def show_convergence(record: ExperimentRecord, orientation = 'v'):

    results = record.get_result()  # type: Duck
    args = record.get_args()

    # newduck = Duck(list(d[['t', 'L(phi_local)', 'L(phi_global)', 'alignments']] for d in results))
    newduck = results.break_in()

    # with vstack_plots(xlabel='Iterations', xlim=(0, args['n_iter']*1.4), left_pad=0.15, bottom_pad=0.1, spacing=0.05):

    plt.figure(figsize=(8, 6) if orientation=='v' else (10, 2))

    context = \
        vstack_plots(xlabel='Iterations', left_pad=0.15, bottom_pad=0.1, right_pad=0.2, spacing=0.05) if orientation =='v' else \
        hstack_plots(xlabel='Iterations', left_pad=0.1, bottom_pad=0.2, right_pad=0.2, spacing=0.4, sharey=False, show_y=True)

    with context:

        select_subplot('a')
        plt.plot(newduck[:, 'Iter'], newduck[:, 'L(phi_local)'], label='$L(\phi_{local})$')
        plt.plot(newduck[:, 'Iter'], newduck[:, 'L(phi_global)'], label='$L(\phi_{global})$')
        plt.grid()
        # plt.legend()
        # outside_right_legend()
        plt.ylabel('Loss')

        select_subplot('b')
        n_layers = args['depth']-1
        for i, c in zip(range(n_layers-1), get_color_cycle_map(name='jet', length=n_layers)):
            # plt.plot(newduck[:, 'Iter'], newduck[:, 'alignments', i], color=c, label=f'$S(\\frac{{\partial L}}{{\partial \phi_{i}}}, \\frac{{\partial L}}{{\partial \phi_{{{i+1}..L}} }})$')
            plt.plot(newduck[:, 'Iter'], newduck[:, 'alignments', i], color=c, label=f'$S(\\nabla_{{\phi_{i+1}}} L_{i+1}, \\nabla_{{\phi_{i+1}}} L_{{{i+2}:{n_layers}}})$')
        plt.ylabel('Gradient Alignment')
        # outside_right_legend()
        plt.grid()

    outside_right_legend(ax=select_subplot('a'))
    outside_right_legend(ax=select_subplot('b'))
    plt.show()



@ExperimentFunction(is_root=True, show=show_convergence)
def demo_local_grad_convergence(
    width,
    depth,
    target_mode,
    nonlinearity = 'tanh',
    init_scale = 1.,
    ortho = False,
    n_samples = 1000,
    n_iter = 5000,
    epsilon_perturb = 0.01,
    minibatch_size = 100,
    learning_rate = 0.001,
    momentum = 0.8
    ):

    layer_sizes = [width]*depth

    x = torch.randn(n_samples, layer_sizes[0])

    phi = [param_init(n_in, n_out, mode='ortho' if ortho else 'xavier', init_scale=init_scale) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]

    if target_mode=='params':
        phi_targ = [param_init(n_in, n_out, mode='ortho' if ortho else 'xavier', init_scale=init_scale) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        states_target = forward_pass(phi=phi_targ, x=x, nonlinearity=nonlinearity)
    elif target_mode=='params-perturb':
        phi_targ = [param_init(n_in, n_out, mode='ortho' if ortho else 'xavier', init_scale=init_scale) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        states_target = [s + epsilon_perturb*torch.randn(s.size()) for s in forward_pass(phi=phi_targ, x=x, nonlinearity=nonlinearity)]
    elif target_mode=='state':
        h = get_named_nonlinearity(nonlinearity)
        states_target = [h(torch.randn(n_samples, k)) for k in layer_sizes[1:]]
    else:
        raise Exception(target_mode)

    phi_local, phi_global = [p.clone() for p in phi], [p.clone() for p in phi]

    # local_optimizer = MomentumOptimizer(learning_rate=learning_rate, momentum=0)
    # global_optimizer = MomentumOptimizer(learning_rate=learning_rate, momentum=0)
    local_optimizer = MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    global_optimizer = MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    # local_optimizer = AdamMaxOptimizer(alpha=0.01)
    # global_optimizer = AdamMaxOptimizer(alpha=0.01)
    # local_optimizer = AdaGradOptimizer(learning_rate=0.01)
    # global_optimizer = AdaGradOptimizer(learning_rate=0.01)

    dbplot(np.array([s.detach().numpy()[:10, :20] for s in states_target]), 'targets')

    results = Duck()


    for t in range(n_iter):

        is_test_iteration = do_every(100, counter_id='test_iter')

        ixs = slice(None) if is_test_iteration else np.random.choice(len(x), size=minibatch_size, replace=False)
        data = x[ixs]
        targets = [st[ixs] for st in states_target]


        states_local = forward_pass(phi=phi_local, x=data, nonlinearity=nonlinearity)
        phi_local_loss_local = [((sf-st)**2).sum(dim=1).mean() for sf, st in izip_equal(states_local, targets)]
        phi_local_grad_local = [torch.autograd.grad(local_loss, inputs = p, retain_graph=True)[0] for p, local_loss in izip_equal(phi_local, phi_local_loss_local)]
        # phi_local = [torch.tensor((p-learning_rate*pg).detach(), requires_grad=True) for p, pg in izip_equal(phi_local, phi_local_grad_local)]

        states_global= forward_pass(phi=phi_global, x=data, nonlinearity=nonlinearity)
        phi_global_loss_global = sum(((sf-st)**2).sum(dim=1).mean() for sf, st in izip_equal(states_global, targets))
        phi_global_grad_global = torch.autograd.grad(phi_global_loss_global, inputs=phi_global, retain_graph=True)
        # phi_global = [torch.tensor((p-learning_rate*pg).detach(), requires_grad=True) for p, pg in izip_equal(phi_global, phi_global_grad_global)]

        phi_local_loss_global = sum(((sf-st)**2).sum(dim=1).mean() for sf, st in izip_equal(states_local, targets))

        label = {'Iter': t, 'L(phi_local)': phi_local_loss_global.item(), 'L(phi_global)': phi_global_loss_global.item()}

        if do_every(20, counter_id=1):
            print(label)
        if is_test_iteration:
            phi_local_grads_global = torch.autograd.grad(phi_local_loss_global, inputs=phi_local, retain_graph=True, allow_unused=True)
            alignments = [cosine_distance(gl, gg-gl, dim=1).mean().item() for gl, gg in izip_equal(phi_local_grad_local, phi_local_grads_global)]

            label['alignments'] = alignments

            local_grad_mags = [abs(gl).mean() for gl in phi_local_grad_local]
            distant_grad_mags = [abs(gl-gg).mean() for gl, gg in izip_equal(phi_local_grad_local, phi_local_grads_global)]

            with hold_dbplots():
                dbplot(np.array([local_grad_mags, distant_grad_mags]).T, 'grad-mags', legend=['local', 'distant'], grid=True)

                print(f'Local-Global Alignments: {alignments}')
                dbplot(np.array(alignments), 'Alignments', plot_type='line', grid=True)
                dbplot(np.array([s.detach().numpy()[:10, :20] for s in states_local]), 'sloc')
                dbplot(np.array([s.detach().numpy()[:10, :20] for s in states_global]), 'sglob')
            results[next, :] = label
            yield results
            with hold_dbplots(draw_every=100):
                dbplot(np.array([phi_local_loss_global.item(), phi_global_loss_global.item()]), 'losses', grid=True, axis='loss', legend=['$L(\phi_{local})$', '$L(\phi_{global})$'])
            # dbplot(phi_global_loss_global, '$L(\phi_{{global}})$', axis='loss')
        else:
            phi_local = local_optimizer.update(params=phi_local, grads=phi_local_grad_local)
            # phi_local = local_optimizer.update(params=phi_local[:1], grads=phi_local_grad_local[:1]) + phi_local[1:]
            phi_global = global_optimizer.update(params=phi_global, grads=phi_global_grad_global)


    # distant_loss_grads = [torch.autograd.grad(sum(losses[i+1:]), inputs = p, retain_graph=True)[0] if i<len(losses)-1 else torch.zeros_like(p) for i, p in enumerate(phi)]




    # # Check that the gradients add up, as expected
    # for loc, dist, glob in izip_equal(local_loss_grads, distant_loss_grads, global_loss_grads):
    #     assert cosine_distance(glob.flatten(), (loc+dist).flatten(), dim=0) > 0.99, 'Gradients not adding up?'
    #     # assert torch.allclose(glob, loc+dist, atol=1e-3, rtol=1e-5) and not torch.allclose(glob, torch.zeros_like(glob))

    # d_local_global = [cosine_distance(gt.flatten(), gl.flatten(), dim=0).mean() for gt, gl in izip_equal(global_loss_grads, local_loss_grads)]
    # d_distant_global = [cosine_distance(gt.flatten(), gl.flatten(), dim=0).mean() for gt, gl in izip_equal(global_loss_grads, distant_loss_grads)]
    # d_local_distant = [cosine_distance(gt.flatten(), gl.flatten(), dim=0).mean() for gt, gl in izip_equal(local_loss_grads, distant_loss_grads)]
    #
    # m_local_over_distant = [(l**2).mean()/(d**2).mean() for l, d in izip_equal(local_loss_grads, distant_loss_grads)]
    # m_local_over_global = [(l**2).mean()/(d**2).mean() for l, d in izip_equal(local_loss_grads, global_loss_grads)]
    #
    # fields = {'D(local,global)': d_local_global, 'D(distant,global)': d_distant_global, 'D(local,distant)': d_local_distant, '||local||/||distant||':m_local_over_distant, '||local||/||global||':m_local_over_global}
    # print(tabulate(build_table(lookup_fcn=lambda row, col: fields[col][row], row_categories=list(range(len(losses))), column_categories=fields.keys(), include_column_category=False), headers=['Depth']+list(fields.keys())))


# ======================================================================================================================

X=demo_local_grad_convergence.add_root_variant('base', width=100, depth=12)
X.add_variant(target_mode='params')
X.add_variant(target_mode='state')
"""
If we optimize only local losses, do we get the the minimum of the global loss when
(A) Targets are generated by a randomly initialized networks?
- Yes, we do - global loss just gets there a bit faster.
(B) Targets are generated by random states?
- Yes - if we use a good optimizer, it seems that the local and global get to the same point.
"""

X=demo_local_grad_convergence.add_root_variant('narrow', width=30, depth=12)
X.add_variant(target_mode='params')
X.add_variant(target_mode='state')


achievable_exp = demo_local_grad_convergence.add_variant('achievable', n_iter = 10000, width=200, depth=8, init_scale=1.6, minibatch_size=100, target_mode='params', ortho=False,  nonlinearity='centered_leaky_clip')
unachievable_exp = demo_local_grad_convergence.add_variant('unachievable', n_iter = 10000, width=200, depth=8, init_scale=1.6, minibatch_size=100, target_mode='state', ortho=False,  nonlinearity='centered_leaky_clip')
achievable_ortho = demo_local_grad_convergence.add_variant('achievable-ortho', n_iter = 10000, width=200, depth=8, init_scale=1.6, minibatch_size=100, target_mode='params', ortho=True,  nonlinearity='centered_leaky_clip')


if __name__ == '__main__':
    demo_local_grad_convergence.browse()
    # achievable_exp.run()
    # unachievable_exp.run()
    # show_convergence(achievable_exp.get_latest_record())
    # show_convergence(unachievable_exp.get_latest_record())
    # show_convergence(achievable_ortho.get_latest_record())

    # achievable_ortho.run()
    # achievable_ortho.browse()

    # demo_local_grad_convergence.call(width=200, depth=8, target_mode='state', ortho=False,  nonlinearity='relu')
    # demo_local_grad_convergence.run(width=200, depth=8, init_scale=1.2, target_mode='params', ortho=False,  nonlinearity='tanh')
    # demo_local_grad_convergence.browse(keep_reco%rd=False)
