from typing import Sequence

from matplotlib import pyplot as plt

from artemis.experiments.experiment_record import ExperimentRecord
from artemis.experiments.experiment_record_view import separate_common_args
from artemis.general.display import dict_to_str
from artemis.general.should_be_builtins import izip_equal
from artemis.plotting.expanding_subplots import select_subplot
from artemis.plotting.pyplot_plus import get_lines_color_cycle
import numpy as np


def report_score_from_result(result):

    if result[-1].has_key('test_init_error'):  # New version
        errors = ('test_init_error', 'test_neg_error', 'train_init_error', 'train_neg_error')
        epoch = result[-1, 'epoch']
        fin_test_init_error, fin_test_neg_error, fin_train_init_error, fin_train_neg_error = (result[-1, err_type] for err_type in errors)
        min_test_init_error, min_test_neg_error, min_train_init_error, min_train_neg_error = (np.min(result[:, err_type]) for err_type in errors)

        return f'Epoch: {epoch:.3g}, ' \
            f'Test Init: {fin_test_init_error:.3g}% m:{min_test_init_error:.3g}%, ' \
            f'Test Neg: {fin_test_neg_error:.3g}% m:{min_test_neg_error:.3g}%,' \
            f'Train Init: {fin_train_init_error:.3g}% m:{min_train_init_error:.3g}%, ' \
            f'Train Neg: {fin_train_neg_error:.3g}% m:{min_train_neg_error:.3g}%'
    else:  # Old version
        epoch, test_error, train_error = result[-1, 'epoch'], result[-1, 'test_error'], result[-1, 'train_error']
        min_test_error = np.min(result[:, "test_error"])
        min_train_error = np.min(result[:, "train_error"])
        return f'Epoch: {epoch:.3g}, Test: {test_error:.3g}% m:{min_test_error:.3g}% Train: {train_error:.3g} m:{min_train_error:.3g}%'


def parse_eqprop_result(result):
    errors = ('test_init_error', 'test_neg_error', 'train_init_error', 'train_neg_error')
    return [('epoch', result[-1, 'epoch'])] + [(err_type, result[-1, err_type]) for err_type in errors] + [(err_type, np.min(result[:, err_type])) for err_type in errors]


def compare_learning_curves(records: Sequence[ExperimentRecord], show_now = True):

    argcommon, argdiffs = separate_common_args(records, as_dicts=True, only_shared_argdiffs=False)
    fig = plt.figure()
    ax = select_subplot(1)
    color_cycle = get_lines_color_cycle()
    for i, (rec, ad, c) in enumerate(zip(records, argdiffs, color_cycle)):
        result = rec.get_result()
        for i, subset in enumerate(('train_error', 'test_error')):
            is_train = subset=="train_error"
            ax.plot(result[:, 'epoch'], result[:, subset], label=('train: ' if subset=='train_error' else 'test: ')+dict_to_str(ad).replace('lambdas', '$\lambda$').replace('epsilon', '$\epsilon$'), linestyle='--' if is_train else '-', alpha=0.7 if is_train else 1, color=c)
            ax.grid()
            ax.legend()
            ax.set_ybound(0, max(10, min(result[:, subset])*1.5))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Error')
    ax.grid()
    if show_now:
        plt.show()


def compare_learning_curves_new(records: Sequence[ExperimentRecord], show_now = True):

    argcommon, argdiffs = separate_common_args(records, as_dicts=True, only_shared_argdiffs=False)
    fig = plt.figure()
    ax = select_subplot(1)
    color_cycle = get_lines_color_cycle()
    subsets = ('test_init_error', 'test_neg_error', 'train_init_error', 'train_neg_error')
    maxminscore = 0
    for i, (rec, ad, c) in enumerate(zip(records, argdiffs, color_cycle)):
        result = rec.get_result()
        for i, (subset, (linestyle, alpha)) in enumerate(izip_equal(subsets, (('-', 1), (':', 1), ('-', .5), (':', .5)))):
            ax.plot(result[:, 'epoch'], result[:, subset], label=('train' if 'train' in subset else 'test')+'-'+('init' if 'init' in subset else 'neg')+': '+dict_to_str(ad).replace('lambdas', '$\lambda$').replace('epsilon', '$\epsilon$'), linestyle=linestyle, alpha=alpha, color=c)
            ax.grid()
            ax.legend()

            maxminscore = max(maxminscore, min(result[:, subset]))

    ax.set_ybound(0, max(10, maxminscore*1.5))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Error')
    ax.grid()
    if show_now:
        plt.show()
