# my logging module
import logging

filename="EXAMPLE.log"; level=logging.DEBUG; format='%(asctime)s %(message)s'
logging.basicConfig(filename=filename, level=level, format=format)
_logger = logging.getLogger()

def energy_term_log(energy_func):
    # decorator for energy function in optimization
    def inner(optimizer, x): # x is expected to be a 1-d numpy array
        res = energy_func(optimizer, x)
        msg = "each terms: "+", ".join([str(y) for y,w,n in res])
        _logger.info(msg)
        return res
    return inner

def energy_sum_log(sum_func):
    # decorator for energy summing function
    def inner(optimizer, x):
        _logger.info("evaluating on x: {x}".format(x=x))
        res = sum_func(optimizer, x)
        _logger.info("total: {res}".format(res=res))
        return res
    return inner

def task_log(run):
    # decorator for run() in optimizer
    def inner(optimizer):
        _logger.info("starting optimization with configuration of:")
        _logger.info("method - {name}".format(name=optimizer._method_name))
        _logger.info("energies:")
        for func, name, weight in optimizer._energy_list:
            # TODO: bad accessing, needs refactoring
            _logger.info("energy function - {name}, weight - {weight}".format(name=name, weight=weight))
        run(optimizer)
        return
    return inner

def init(filename="EXAMPLE.log", level=logging.DEBUG, format='%(asctime)s %(message)s'):
    _file_name=filename
    logging.basicConfig(filename=filename, level=level, format=format)
    global _logger
    _logger = logging.getLogger()

# all these customized wrapping-capable loggers shall be singletons
class Energy_sum_logger(logging.Logger):
    def __init__(self):
        pass
    
    # called as a decorator
    def __call__(self, f):
        return f
    
    # TODO: provide interface to modify configuration at runtime


class Energy_term_logger(logging.Logger):
    def __init__(self):
        pass
    
    # called as a decorator
    def __call__(self, f):
        return f
    
    # TODO: provide interface to modify configuration at runtime

class Task_logger(logging.Logger):
    def __init__(self):
        pass
    
    # called as a decorator
    def __call__(self, f):
        return f
    
    # TODO: provide interface to modify configuration at runtime
