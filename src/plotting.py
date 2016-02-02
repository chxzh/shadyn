import matplotlib.pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle
class Plotter():
    def __init__(self, file_name):
        self._eval_term_records = ()
        self._eval_sum_records = None
        self._fname = file_name
    
    def set_param(self, func_names):
        self._term_names = func_names[:]
        self._eval_term_records = tuple([] for name in func_names)
        self._eval_term_records = []
   
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
    
