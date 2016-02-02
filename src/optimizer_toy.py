import log
class Optimizer:
    def __init__(self):
        pass
    
    def _sum(f):
        @log.energy_sum_log
        def inner(cls, x):
            res = f(cls, x)
            return sum([w*i for i,w,n in res])
        return inner
    
    @_sum
    @log.energy_term_log
    def energy_func(self, x):
        return [(i+x, i-0.5, str(i)) for i in xrange(3)]
    
    @log.task_log
    def run(self):
        for i in xrange(5):
            self.energy_func(i)
    
    def __repr__(self):
        return 'optimizer'
    