# my logging module
import logging
from functools import wraps
from astropy.logger import Logger

def init(filename="EXAMPLE.log", level=logging.DEBUG, format='%(asctime)s %(message)s'):
    _file_name=filename
    hdl = logging.FileHandler(filename)
    fmt = logging.Formatter(format)
    hdl.setFormatter(fmt)
    global energy_sum_log, energy_term_log, task_log
    for logger in [energy_sum_log, energy_term_log, task_log]:
        logger.addHandler(hdl)

# all these customized wrapping-capable loggers shall be singletons
class Energy_sum_logger(logging.Logger):
    def __init__(self, *args, **kwds):
        logging.Logger.__init__(self, *args, **kwds)
    
    # called as a decorator for energy summing function
    def __call__(self, get_energy):
        @wraps(get_energy)
        def wrapper(optimizer, x):
            self.log(self.level, "evaluating on x: {x}".format(x=x))
            res = get_energy(optimizer, x)
            self.log(self.level, "total: {res}".format(res=res))
            return res
        return wrapper
    
    # TODO: provide interface to modify configuration at runtime


class Energy_term_logger(logging.Logger):
    def __init__(self, *args, **kwds):
        logging.Logger.__init__(self, *args, **kwds)
    
    # called as a decorator for energy function in optimization
    def __call__(self, get_energy_terms):
        @wraps(get_energy_terms)
        def wrapper(*args, **kwds):
            res = get_energy_terms(*args, **kwds)
            msg = "each terms: "+", ".join([str(y) for y,w,n in res])
            self.log(self.level, msg)
            return res
        return wrapper
    
    # TODO: provide interface to modify configuration at runtime

class Task_logger(logging.Logger):
    def __init__(self, *args, **kwds):
        logging.Logger.__init__(self, *args, **kwds)
    
    # called as a decorator of running an optimization task
    def __call__(self, run):
        @wraps(run)
        def wrapper(optimizer, *args, **kwds):
            # optimizer is the bounded instance of run()
            self.log(self.level, "starting optimization with configuration of:")
            self.log(self.level, "method - {name}".format(
                                    name=optimizer._method_name))
            self.log(self.level, "energies:")
            for func, name, weight in optimizer._energy_list:
                # TODO: bad accessing, needs refactoring
                self.log(self.level,
                            "energy function - {name}, weight - {weight}".format(
                                name=name, weight=weight))
            res = run(optimizer, *args, **kwds) # i.e. optimizer.run(*args, **kwds)
            return res
        return wrapper
    
    # TODO: provide interface to modify configuration at runtime


energy_term_log = Energy_term_logger("energy term logger", logging.INFO)
energy_sum_log = Energy_sum_logger("energy sum logger", logging.INFO)
task_log = Task_logger("task logger", logging.INFO)
init()

