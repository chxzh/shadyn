# my logging module
import logging
_file_name = ""
logging.basicConfig()
_logger = None

def energy_term_log(energy_func):
    # decorator for energy function in optimization
    def inner(optimizer, x): # x is expected to be a 1-d numpy array
        res = energy_func(x)
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

def init(filename, level=logging.DEBUG):
    _file_name=filename
    logging.basicConfig(filename=filename, level=level)
    _logger = logging.getLogger(name)
