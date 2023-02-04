# -*- coding: utf-8 -*-

from ushap import ShapleyValues
from tools import extract_from_circuit


class QuantumShapleyValues(ShapleyValues):
    def __init__(self, qc, value_fun, value_kwargs_dict, quantum_instance, shap_sample_frac=None, shap_sample_reps=1, evaluate_value_only_once=False, shap_sample_seed=None, shap_batch_size=None, qc_preprocessing_fun=None, locked_instructions=None, memory=None, callback=None, delta_exponent=1, name=None, silent=False):
        '''
        Parameters
        ----------
        qc : qiskit.circuit.QuantumCircuit
            Quantum circuit of interest. Per default, all gates of the circuit are used as players if not specified as locked_instructions.
        value_fun : callable
            Value function, i.e., function to map circuits to floats: (qc_data, num_qubits, S, quantum_instance, **kwargs) -> (value)
        value_kwargs_dict : dict
            Dictionary to be passed to value_fun for every evaluation.
        quantum_instance : qiskit.utils.QuantumInstance
            Quantum instance to be passed to value_fun for every evaluation.
        shap_sample_frac : float (>0) or None or int (<0)  (default: None)
            Fraction of coalitions that is considered for each player:
                a) None: no subsampling. Consider all 2**N coalitions.
                b) positive float: subsampling, sample shap_sample_frac*100% of all 2**(N-1) possible coalitions (can be > 1).
                c) negative integer: subsampling, sample abs(shap_sample_frac) of all 2**(N-1) possible coalitions (can be > 2**N).
        shap_sample_reps : int or none (default: 1)
            Number of repeated evaluations for each value function. For each considered coalition, 2*shap_sample_reps value functions are calculated. The mean of all value functions for the same coalition is used to determine the Shapley values.
        evaluate_value_only_once : bool (default: False)
            If true, evaluate every value function only once and recall from memory afterwards. Otherwise, allows to evaluate each value function multiple times. Is required to be false for shap_sample_reps>1 to have any effect.
        shap_sample_seed : int or None, optional (default: None)
            Random seed for numpy.random.RandomState, used for subsampling.
        shap_batch_size : int or None, optional (default: None)
            If shap_batch_size is not None, multiple coalitions (up to shap_batch_size) are evaluated at once. This can be useful to submit multiple circuits to a backend at once.
            Accordingly, value_fun is expected to be of the form: (qc_data, num_qubits, S_list, quantum_instance, **kwargs) -> (value_list)
        qc_preprocessing_fun : callable, optional (default: None)
            Preprocessing function for the quantum circuit: (qc) -> (qc), is ignored if None.
        locked_instructions : list, optional (default: None)
            Gate indices of the circuit that are always activated and do not act as players for Shapley values. None corresponds to [], i.e., all gates are players.
        memory : dict, optional (default: None)
            Dictionary to recall the memory from previous calculations from (see memory property). Is also used to store new calculations. None corresponds to a fresh/blank memory.
        callback : callable, optional (default: None)
            Function to be called before every value function evaluation: (S) -> None, is ignored if None.
        delta_exponent : int, optional (default: 1)
            Can be used to calculate higher moments, see d_calculation. For Shapley values, use 1.
        name : str, optional (default: None)
            Only for displaying purposes. Defaults to a standard name for None.
        silent : bool, optional (default: False)
            If True, hide progess bars.           
        '''
        
        # preprocess
        self._qc_preprocessing_fun = qc_preprocessing_fun
        if self._qc_preprocessing_fun is not None:
            qc = self._qc_preprocessing_fun(qc)
        self._num_qubits, self._qc_data = extract_from_circuit(qc, locked_instructions)
        unlocked_instructions = [idx for idx, (instr, qargs, cargs, opts) in enumerate(self._qc_data) if not opts['lock']]
            
        # setup value function kwargs
        self._quantum_instance = quantum_instance
        effective_value_kwargs_dict = {'qc_data': self._qc_data, 'num_qubits': self._num_qubits, 'quantum_instance': self._quantum_instance}
        effective_value_kwargs_dict.update(value_kwargs_dict)
         
        # initialize
        super().__init__(unlocked_instructions, locked_instructions, value_fun, effective_value_kwargs_dict, shap_sample_frac, shap_sample_reps, shap_batch_size, evaluate_value_only_once, shap_sample_seed, memory, callback, delta_exponent, name, silent)
        
    def run(self):
        '''
        Evaluate Shapley values.

        Returns
        -------
        phi_dict : dict
            Dictionary of Shapley values of the form {player index: value, ...}.
            Result is also stored in phi_dict property.

        '''
        return self()
    
    def get_values(self, S_list, recall=False):
        '''
        Evaluate value functions.
        
        Parameters
        ----------
        S_list : list
            List of coalitions to evaluate (i.e., a list of lists) .
        recall : bool, optional (default: False)
            If true, recall value function from memory.

        Returns
        -------
        values : list
            List of values, one float for every coalition.

        '''
        return self.eval_S_list(S_list, recall)

    def disp(self):
        '''
        Print settings.

        Returns
        -------
        None.

        '''
        print(self.__str__())
        
    def get_summary_dict(self, property_list=[]):
        '''
        Return a summary of the most important properties in form of a dictionary.
        
        Parameters
        ----------
        property_list : list, optional (default: [])
            List of property names to additionally include in the summary.

        Returns
        -------
        summary : dict
            Dictionary containing selected properties.
            
        '''
        def get_attr(name):
            return getattr(self, name) if hasattr(self, name) else None
        summary = super().get_summary_dict(property_list)
        summary.update({'quantum_instance': get_attr('_quantum_instance'),
                        #'qc': get_attr('_qc'),
                        'qc_preprocessing_fun': get_attr('_qc_preprocessing_fun'),
                        #'qc_data': get_attr('_qc_data'),
                        'num_qubits': get_attr('_num_qubits')})
        return summary  
    