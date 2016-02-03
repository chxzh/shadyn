import matplotlib.pyplot as plt
from functools import wraps
from IPython.core.magic_arguments import kwds
try:
   import cPickle as pickle
except:
   import pickle

PLOT_ENABLED = True

class Plotter():
    def __init__(self, file_name):
        self._eval_term_records = ()
        self._eval_sum_records = None
        self._fname = file_name
    
    def set_param(self, func_names):
        self._term_names = func_names[:]
        self._eval_term_records = tuple([] for name in func_names)
        self._eval_sum_records = []
   
    def record_terms(self, ywn_shashlik):
        for record, ywn in zip(self._eval_term_records, ywn_shashlik):
            y, w, n = ywn
            record.append(y)
    
    def record_sum(self, y):
        self._eval_sum_records.append(y)
    
    def plot_result(self):
        # pickle the result for later availability
        with open(self._fname+'.terms', 'wb') as handle:
            pickle.dump(self._eval_term_records, handle)
        with open(self._fname+'.sums', 'wb') as handle:
            pickle.dump(self._eval_sum_records, handle)
        # plot the things
        plt.plot(self._eval_sum_records, label="sum")
        for term, name in zip(self._eval_term_records, self._term_names):
            plt.plot(term, label=name)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
        plt.savefig(self._fname+'.png')


def conditional(condition):
    """Meta-decorator that enables the decorated decorator only when
    the given condition is met.
    """
    def decorate(other_dec):
        if condition:
            return other_dec
        else:
            def empty_decorator(decorated):
                return decorated
            return empty_decorator
    return decorate

@conditional(PLOT_ENABLED)
def plot_terms(f):
    """Decorator for energy (terms) function of optimizer, records terms of
    every evaluation to plot"""
    # needs to inject a plotter attribute into Optimizer instance
    @wraps(f)
    def inner(cls, *args, **kwds):
        res = f(cls, *args, **kwds)
        return res
    return inner

@conditional(PLOT_ENABLED)
def plot_sum(f):
    """Decorator for sum of energy (sum) function of optimizer, records the sum
    of all terms of each evaluation"""
    # needs a way to inject an attribute of plotter into the Optimizer instance
    @wraps(f)
    def inner(cls, *args, **kwds):
        y = f(cls, *args, **kwds)
        cls.plotter.record_sum(y)
        return y
    return inner

@conditional(PLOT_ENABLED)
def plot_task(f):
    """Decorator for run() of optimizer, plot and save the things"""
    @wraps(f)
    def inner(cls, *args, **kwds):
        res = f(cls, *args, **kwds)
        cls.plotter.plot_result()
        return res
    return inner

@conditional(PLOT_ENABLED)
def init_plot(set_energy):
    """Decorator for set_energy() of optimizer, initialize the records"""
    @wraps(set_energy)
    def inner(cls, func_names, weights):
        cls.plotter.set_param(func_names)
        res = set_energy(cls, func_names, weights)
        return res
    return inner
     
def attach_plotter(optimizer, plotter):
    if PLOT_ENABLED: setattr(optimizer, 'plotter', plotter)
    optimizer.plotter.set_param






