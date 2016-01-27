# my logging module
import logging
logging.basicConfig()
_logger = logging.getLogger(name)

def energy_log(energy_func):
    # decorator for energy function in optimization
    def inner(optimizer, x): # x is expected to be a 1-d numpy array
        res = energy_func(x)
        _logger.info("evaluating on x:{x} -> {}".format(x=x))
        return res
    return inner

def optimtask_log(run):
    # decorator for run() in optimizer
    def inner(optimizer):
        _logger.info("starting optimization with configuration of:")
        _logger.info("method - {name}", name=optimizer._method_name)
        _logger.info("energies:")
        for func, name, weight in optimizer._energy_list:
            # TODO: bad accessing, needs refactoring
            _logger.info("energy function - {name}, weight - {weight}",
                         name=name, weight=weight)
        run()
        return
    return inner