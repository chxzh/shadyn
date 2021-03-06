import matplotlib.pyplot as plt
from functools import wraps
from IPython.core.magic_arguments import kwds
from _imaging import path
import numpy as np
import os, traceback
try:
   import cPickle as pickle
except:
   import pickle

PLOT_ENABLED = True

class Plotter():
    def __init__(self, path, time_stamp):
        self._eval_term_records = ()
        self._eval_sum_records = None
        self._x_records = None
        self._path = path
        self._time_stamp = time_stamp
        self._fname = path+'\\'+time_stamp
        self._method_name = None
    
    def set_method(self, method_name):
        self._method_name = method_name
    
    def set_param(self, func_names):
        self._term_names = func_names[:]
        self._eval_term_records = tuple([] for name in func_names)
        self._eval_sum_records = []
        self._x_records = []
   
    def record_terms(self, ywn_shashlik):
        for record, ywn in zip(self._eval_term_records, ywn_shashlik):
            y, w, n = ywn
            record.append(y)
    
    def record_sum(self, y):
        self._eval_sum_records.append(y)
    
    def record_x(self, x):
        self._x_records.append(x)
    
    def plot_result(self):
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        # pickle the result for later availability
        with open(self._fname+'.terms', 'wb') as handle:
            pickle.dump(self._eval_term_records, handle)
        with open(self._fname+'.sums', 'wb') as handle:
            pickle.dump(self._eval_sum_records, handle)
        with open(self._fname+'.xs', 'wb') as handle:
            pickle.dump(self._x_records, handle)
        # plot the things
        
        plt.figure(figsize=(16, 12))
        ax = plt.subplot(111)  
        plt.plot(np.log10(self._eval_sum_records), label="total")
        for term, name in zip(self._eval_term_records, self._term_names):
            ax.plot(np.log10(term), label=name)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - box.height * 0.05,
                         box.width, box.height*0.95])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                  fancybox=True, shadow=True, ncol=2)
        plt.title(self._method_name, y=-0.1)
        plt.savefig(self._path+"\\..\\"+self._time_stamp+'.png')
        plt.savefig(self._fname+'.png')
        plt.close()
        
    


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
        cls.plotter.record_terms(res)
        return res
    return inner

@conditional(PLOT_ENABLED)
def plot_sum(f):
    """Decorator for sum of energy (sum) function of optimizer, records the sum
    of all terms of each evaluation"""
    # needs a way to inject an attribute of plotter into the Optimizer instance
    @wraps(f)
    def inner(cls, x):
        y = f(cls, x)
        cls.plotter.record_x(x)
        cls.plotter.record_sum(y)
        return y
    return inner

@conditional(PLOT_ENABLED)
def plot_task(f):
    """Decorator for run() of optimizer, plot and save the things"""
    @wraps(f)
    def inner(cls, *args, **kwds):
        try:
            res = f(cls, *args, **kwds)
            return res
        except:
            traceback.print_last()
        finally:
            
            cls.plotter.set_method(cls._method_name) # TODO: terrible accessing
            cls.plotter.plot_result()
            cls.renderer.set_param(cls.best_x)
            # TODO: how to deal with atb parameters unmatch
            try:
                img = cls.renderer.acquire_snapshot()
            except RuntimeError as e:
                print e.message
                cls.renderer = cls.rendis.acquire_new()
                img = cls.renderer.acquire_snapshot()
                
            img.save(cls.plotter._path + "\\final_result_snapshot.png") # TODO: ehh
            img = cls.renderer.acquire_full_snapshot()
            img.save(cls.plotter._path + "\\final_full_snapshot.png") # TODO: ehh II

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







