# -*- coding: utf-8 -*-

from ushap import ShapleyValues


class ClassicalShapleyValues(ShapleyValues):
    def __init__(self, N, value_fun, value_kwargs_dict={}, shap_sample_frac=None, shap_sample_reps=1, evaluate_value_only_once=False, shap_sample_seed=None, shap_batch_size=None, memory=None, callback=None, delta_exponent=1, name=None, silent=False):         
        # process options
        self._N = int(N)
        locked_instructions = []
        unlocked_instructions = list(range(self._N))
         
        # initialize
        super().__init__(unlocked_instructions, locked_instructions, value_fun, value_kwargs_dict, shap_sample_frac, shap_sample_reps, shap_batch_size, evaluate_value_only_once, shap_sample_seed, memory, callback, delta_exponent, name, silent)
        
    def get_summary_dict(self, property_list=[]):
        def get_attr(name):
            return getattr(self, name) if hasattr(self, name) else None
        summary = super().get_summary_dict(property_list)
        summary.update({'N': get_attr('_N')})
        return summary          
    