# my logging module
import logging
logging.basicConfig()
_logger = logging.getLogger(name)

def energy_log(energy):
    # decorator for energy function in optimization
    def inner(self, x):