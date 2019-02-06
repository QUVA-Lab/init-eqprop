from matplotlib import pyplot as plt

from artemis.experiments.experiment_record import ExperimentRecord
from artemis.experiments.experiment_record_view import make_record_comparison_duck
from artemis.experiments.experiments import Experiment
from artemis.general.duck import Duck
from artemis.plotting.expanding_subplots import subplot_at, set_figure_border_size, add_subplot, vstack_plots, \
    hstack_plots
from artemis.plotting.pyplot_plus import get_color_cycle_map, modify_color


def plot_experiment_result(record: ExperimentRecord, field, **kwargs):
    result = record.get_result()
    epochs = result[:, 'epoch'].to_array()
    scores = result[:, field].to_array()
    h, = plt.plot(epochs, scores, **kwargs)
    return h, result


def figure_mnist_eqprop_along():

    from init_eqprop.demo_energy_eq_prop_A_initialized import demo_energy_based_initialize_eq_prop

    random_flip_beta = True
    forward_nonlinearity = 'rho(x)+0.01*x'
    baseline = demo_energy_based_initialize_eq_prop.get_variant('losstype_baseline')

    result_eqprop = baseline.get_variant(random_flip_beta=random_flip_beta, forward_nonlinearity=forward_nonlinearity, train_with_forward=False).get_latest_record().get_result()
    result_initialized = baseline.get_variant(random_flip_beta=random_flip_beta, forward_nonlinearity=forward_nonlinearity, train_with_forward='contrast').get_latest_record().get_result()

    eqprop_test = result_eqprop[:, 'test_neg_error'].to_array()
    init_fwd_test = result_initialized[:, 'test_init_error'].to_array()
    init_neg_test = result_initialized[:, 'test_neg_error'].to_array()

    plt.figure(figsize=(6, 4))
    plt.plot(eqprop_test, 'C0', linewidth=2, label='Eq Prop: $s^-$')
    plt.plot(init_fwd_test, 'C1', linewidth=2, linestyle = '--', label='Init Eq Prop: $s^f$')
    plt.plot(init_neg_test, 'C1', linewidth=2, label = 'Init Eq Prop: $s^-$')
    plt.ylabel('Classification Error (Test)')
    plt.xlabel('Epoch')
    plt.ylim(0, 6)
    plt.legend()
    plt.grid()
    plt.show()


def figure_local_vs_full():

    from init_eqprop.demo_energy_eq_prop_A_initialized import demo_energy_based_initialize_eq_prop

    ex_local = demo_energy_based_initialize_eq_prop.get_variant('local_loss_baseline').get_variant(local_loss=True)
    ex_global = demo_energy_based_initialize_eq_prop.get_variant('local_loss_baseline').get_variant(local_loss=False)
    plt.figure(figsize=(6, 3))
    set_figure_border_size(border=0.1, bottom=0.15)
    plot_experiment_result(ex_local.get_latest_record(if_none='run'), field='test_init_error', color='C2', label='Local: s^f')
    plot_experiment_result(ex_local.get_latest_record(if_none='run'), field='test_neg_error', color='C1', label='Local: s^-')
    plot_experiment_result(ex_global.get_latest_record(if_none='run'), field='test_init_error', color='C2', linestyle='--', label='Global: s^f')
    plot_experiment_result(ex_global.get_latest_record(if_none='run'), field='test_neg_error', color='C1', linestyle='--', label='Global: s^i')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Test Error')
    plt.legend()
    plt.ylim(0, 10)
    plt.grid()
    plt.show()


def figure_mnist_eqprop_multi_size():

    forward_deviation_cost = 0.1
    from init_eqprop.demo_energy_eq_prop_AE_initialized_fwd_energy import baseline_final_small, baseline_final_large

    plt.figure(figsize=(7, 4))
    set_figure_border_size(wspace=0.05, hspace=0.2, bottom=0.11, top=0.07, left=0.1)
    for j, size_baseline in enumerate((baseline_final_small, baseline_final_large)):
        # for i, n_negative_steps in enumerate((10, 20) if size_baseline==baseline_final_small else (20, 50)):
        for i, n_negative_steps in enumerate((4, 20) if size_baseline==baseline_final_small else (20, 50)):
            ax=subplot_at(i, j)
            h_eqprop, result_eqprop = plot_experiment_result(size_baseline.get_variant(train_with_forward=False, n_negative_steps = n_negative_steps, forward_deviation_cost=0).get_latest_record(if_none='run'), 'test_neg_error', label='Eq Prop: $s^-$')
            h_init, result_init = plot_experiment_result(size_baseline.get_variant(train_with_forward='contrast', n_negative_steps = n_negative_steps, forward_deviation_cost=forward_deviation_cost).get_latest_record(if_none='run'), 'test_init_error', label='Init Eq Prop: $s^f$', color='C2')
            h_neg, _ = plot_experiment_result(size_baseline.get_variant(train_with_forward='contrast', n_negative_steps = n_negative_steps, forward_deviation_cost=forward_deviation_cost).get_latest_record(if_none='run'), 'test_neg_error', label='Init Eq Prop: $s^-$', color='C1')

            print(f'Final Scores for {size_baseline.name[size_baseline.name.rfind(".")+1:]}, T={n_negative_steps}: Eqprop {result_eqprop[-1, "test_neg_error"]:.3g}, {result_init[-1, "test_init_error"]:.3g}, {result_init[-1, "test_neg_error"]:.3g}')

            plt.ylim(0, 10 if i>0 else 30)

            if i==0:
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*3.3/10, s=f'{result_eqprop[-1, "test_neg_error"]:.3g}', color=h_eqprop.get_color(), )
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*2.3/10, s=f'{result_init[-1, "test_init_error"]:.3g}', color=h_init.get_color(), )
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*1.3/10, s=f'{result_init[-1, "test_neg_error"]:.3g}', color=h_neg.get_color(), )
            else:
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*5.3/10, s=f'{result_eqprop[-1, "test_neg_error"]:.3g}', color=h_eqprop.get_color(), )
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*4.3/10, s=f'{result_init[-1, "test_init_error"]:.3g}', color=h_init.get_color(), )
                plt.text(x=result_init[-1, 'epoch']*4/5, y=ax.get_ylim()[1]*3.3/10, s=f'{result_init[-1, "test_neg_error"]:.3g}', color=h_neg.get_color(), )

            plt.grid()
            plt.title(f'[784-{"500" if  size_baseline==baseline_final_small else "500-500-500"}-10], {n_negative_steps}-step')
            if i<1:
                ax.xaxis.set_ticklabels([])
            else:
                plt.xlabel('Epoch')

            if j>0:
                ax.yaxis.set_ticklabels([])
            else:
                plt.ylabel('Classification Test Error')

            if i==j==0:
                plt.legend()

    plt.show()


def figure_compare_lambdas():

    from init_eqprop.demo_energy_eq_prop_AE_initialized_fwd_energy import baseline_final_final
    records = [r for r in baseline_final_final.get_variant_records(only_last=True, only_completed=True, ).values() if r is not None]

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


def figure_alignment():

    from init_eqprop.grad_convergence_experiment_D_convergence import achievable_exp, show_convergence
    show_convergence(achievable_exp.get_latest_record(if_none='run'), orientation='h')


def figure_alignment_during_training(orientation='h'):

    from init_eqprop.demo_energy_eq_prop_AH_grad_alignment import ex_large_alignment

    result_local = ex_large_alignment.get_latest_record(if_none='run').get_result()  # type: Duck
    result_global = ex_large_alignment.get_variant(local_loss=False).get_latest_record(if_none='run').get_result()  # type: Duck

    alignments = result_local[:, 'alignments'].break_in().to_array()[:, :-1]

    length = min(len(result_local), len(result_global))
    plt.figure(figsize=(8, 6) if orientation=='v' else (10, 2.3))

    context = \
        vstack_plots(grid=True, xlabel='Epochs', left_pad=0.15, bottom_pad=0.1, right_pad=0.1, spacing=0.05) if orientation =='v' else \
        hstack_plots(grid=True, xlabel='Epochs', left_pad=0.1, bottom_pad=0.2, right_pad=0.1, spacing=0.4, sharey=False, show_y=True)

    with context:
        add_subplot()
        plt.plot(result_local[:length, 'epoch'], result_local[:length, 'test_init_error'], label='Local $s^f$', color='C0')
        plt.plot(result_local[:length, 'epoch'], result_local[:length, 'test_neg_error'], label='Local $s^-$', color=modify_color('C0', modifier=None), linestyle=':')
        plt.plot(result_global[:length, 'epoch'], result_global[:length, 'test_init_error'], label='Global $s^f$', color='C1')
        plt.plot(result_global[:length, 'epoch'], result_global[:length, 'test_neg_error'], label='Global $s^-$', color=modify_color('C1', modifier=None), linestyle=':')
        plt.ylim(0, 10)
        plt.legend(loc = 'Upper Right')
        plt.ylabel('Test Classification Error')

        add_subplot()
        for i, c in zip(range(alignments.shape[1]), get_color_cycle_map('jet', length=alignments.shape[1]+4)):
            plt.plot(result_local[:length, 'epoch'], alignments[:length, i], color=c, label=f'$S(\\nabla_{{\phi_{i+1}}} L_{i+1}, \\nabla_{{\phi_{i+1}}} L_{{{i+2}:{alignments.shape[1]+1}}})$')
        plt.ylabel('Alignment')
        plt.legend()
    plt.show()


if __name__ == '__main__':

    figs = {1: figure_alignment, 2: figure_mnist_eqprop_multi_size, 3: figure_alignment_during_training, 4: figure_compare_lambdas}

    print('This program regenerates the figures from "Initialized Equilibrium Propagation for Backprop-Free Training"')
    print('The first time you run, it may take a while because it computes the figure from scratch.  After that it should \n just load the figure from the saved data.')

    figs[int(input(f'Which figure from the paper would you like to create?  (1-{len(figs)}) >> '))]()
