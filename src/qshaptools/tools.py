import math
from itertools import chain, combinations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit


def p_coalition(coalition_len, total_len):
    p = math.factorial(coalition_len) * math.factorial(total_len - coalition_len - 1) / math.factorial(total_len)
    return p


def powerset(iterable):
    s = list(iterable)
    P_length = 2 ** (len(s))
    P = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return P, P_length


def build_circuit(qc_data, num_qubits, S=None, cl_bits=True):
    qc = QuantumCircuit(num_qubits, num_qubits if cl_bits else 0)
    param_def_dict = {}
    for idx, qc_data_iter in enumerate(qc_data):
        if S is None or idx in S:
            try:
                (instr, qargs, cargs, opts) = qc_data_iter
            except:
                (instr, qargs, cargs) = qc_data_iter
            # 
            # name = instr.name
            # params = [param for param in instr.params]
            qubits = [qubit.index for qubit in qargs]
            clbits = [clbit.index for clbit in cargs]
            qc.append(instr, qubits, clbits)
            # getattr(qc, name)(*params, *qubits, *clbits)
        for param in qc.parameters:
            param_def_dict[param] = None
    return qc, param_def_dict


def extract_from_circuit(qc, locked_instructions):
    num_qubits = qc.num_qubits
    qc_data = []
    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        opts = {}
        if locked_instructions is not None and idx in locked_instructions:
            opts['lock'] = True
        else:
            opts['lock'] = False
        qc_data.append((instr, qargs, cargs, opts))
    return num_qubits, qc_data


def evaluate_circuit(qc, param_def_dict, quantum_instance, counts, sv, add_measurement=True):
    qc_list = [qc]
    param_def_dict_list = [param_def_dict]
    results = evaluate_circuits(qc_list, param_def_dict_list, quantum_instance, counts, sv, add_measurement)
    if counts and sv:
        counts_list, sv_list = results
        return counts[0], sv[0]
    return results[0]


def evaluate_circuits(qc_list, param_def_dict_list, quantum_instance, counts, sv, add_measurement=True):
    for idx in range(len(qc_list)):
        qc = qc_list[idx]
        param_def_dict = param_def_dict_list[idx]
        qc = qc.copy().assign_parameters(param_def_dict)
        if add_measurement:
            qc.measure(range(qc.num_qubits), range(qc.num_qubits))
        qc_list[idx] = qc
    result = quantum_instance.execute(qc_list)
    if counts:
        counts_list = [result.get_counts(qc) for qc in qc_list]
    if sv:
        sv_list = [result.get_statevector(qc) for qc in qc_list]
    if counts and sv:
        return counts_list, sv_list
    elif counts:
        return counts_list
    elif sv:
        return sv_list
    else:
        return [None for qc in qc_list]


def unbind_parameters(qc, name='theta'):
    qc = qc.copy()
    pqc = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    pvec = ParameterVector(name, 0)
    for instr, qargs, cargs in qc:
        instr = instr.copy()
        if instr.params:
            num_params = len(instr.params)
            pvec.resize(len(pvec) + num_params)
            instr.params = pvec[-num_params:]
        pqc.append(instr, qargs, cargs)
    return pqc


def merge_circuit_instructions(qc, merge_instructions_list, names_list=None):
    # check args
    l = np.array(list(chain.from_iterable(merge_instructions_list))).ravel()  # flat list
    assert all(l[i] <= l[i + 1] for i in range(len(l) - 1)), 'instructions unsorted'
    assert all([i in l for i in range(max(l) + 1)]), 'instructions left out'
    assert all([i in l for i in range(len(qc.data))]), 'instructions missing'
    assert names_list is None or len(names_list) == len(merge_instructions_list), 'invalid names'

    # merge
    num_qubits = qc.num_qubits
    qc_merged = QuantumCircuit(num_qubits, 0)
    for merge_idx, merge_instructions in enumerate(merge_instructions_list):
        if len(merge_instructions) == 1:
            (instr, qubits, clbits) = qc.data[merge_instructions[0]]
            qc_merged.append(instr, qubits, clbits)
        else:
            merge_data = [qc.data[idx] for idx in merge_instructions]
            qc_sub = QuantumCircuit(num_qubits, 0)
            qubits_all = []
            clbits_all = []
            names = []
            for (instr, qargs, cargs) in merge_data:
                qubits = [qubit.index for qubit in qargs]
                clbits = [clbit.index for clbit in cargs]
                qubits_all.extend(qubits)
                clbits_all.extend(clbits)
                names.append(instr.name)
                qc_sub.append(instr, qubits, clbits)
            qubits_all = list(set(qubits_all))
            clbits_all = list(set(clbits_all))
            instr = qc_sub.to_instruction()
            if names_list is None or names_list[merge_idx] is None:
                name = '(' + '@'.join(names) + ')'
            else:
                name = names_list[merge_idx]
            instr.name = name
            qc_merged.append(instr, qubits_all, clbits_all)
    return qc_merged


def filter_instructions_by_name(qc_data, filter_fun):
    filtered_idx = []
    for idx, (instr, qargs, cargs) in enumerate(qc_data):
        if filter_fun(instr.name):
            filtered_idx.append(idx)
    return filtered_idx


def remove_instructions_from_circuit(qc, allowed_idx_list):
    qc = qc.copy()
    if allowed_idx_list is not None:
        qc.data = [g for idx, g in enumerate(qc.data) if idx in allowed_idx_list]
    return qc


def visualize_shapleys(qc, phi_dict=None, label_fun=None, digits=2, max_param_str=0, **kwargs):
    if phi_dict is None:
        digits = 0
    if label_fun is None:
        def label_fun(phi_, name_str_, params_str_, digits_, **kwargs):
            return f'{name_str_}{params_str_}:{phi_:+.{digits_}f}'
    qc_vis = qc.copy()
    for i in range(len(qc_vis.data)):
        if phi_dict is None or i in phi_dict:
            if phi_dict is not None:
                phi = phi_dict[i]
            else:
                phi = i
            if len(qc_vis.data[i][0]._params) > 0 and max_param_str > 0:
                params_str = ','.join([str(p) for p in qc_vis.data[i][0]._params])
                params_str = '(' + params_str[:max_param_str] + ('...' if len(params_str) > max_param_str else '') + ')'
            else:
                params_str = ''
            name_str = qc_vis.data[i][0].name
            qc_vis.data[i][0]._label = label_fun(phi, name_str, params_str, digits, **kwargs)
            qc_vis.data[i][0]._params = []
    return qc_vis
