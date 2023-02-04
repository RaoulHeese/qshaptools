from itertools import product
import numpy as np
from qiskit.opflow import AbelianGrouper
from qiskit.quantum_info import Statevector
from scipy.stats import entropy
from tools import build_circuit, evaluate_circuit, evaluate_circuits
from values import value_fun_batch_wrapper_base


def value_callable(qc_data, num_qubits, S, quantum_instance, eval_fun, **eval_fun_kwargs):
    # value: from callable of the form eval_fun(quantum_instance, qc, param_def_dict, **eval_fun_kwargs)
    qc, param_def_dict = build_circuit(qc_data, num_qubits, S)

    # evaluate and return
    value = eval_fun(quantum_instance, qc, param_def_dict, **eval_fun_kwargs)
    return value


def value_fun_batch_wrapper(qc_data, num_qubits, S_list, quantum_instance, wrapped_value_fun, **kwargs):
    # for shapley_iteration_batch
    return value_fun_batch_wrapper_base(S_list, wrapped_value_fun, qc_data=qc_data, num_qubits=num_qubits,
                                        quantum_instance=quantum_instance, **kwargs)


def value_batch_callable(qc_data, num_qubits, S_list, quantum_instance, eval_batch_fun, **eval_fun_kwargs):
    # for shapley_iteration_batch
    # process S_list
    args_list = [build_circuit(qc_data, num_qubits, S) for S in S_list]  # list of (qc, param_def_dict)

    # evaluate and return
    value_list = eval_batch_fun(quantum_instance, args_list, **eval_fun_kwargs)
    return value_list


def value_H(qc_data, num_qubits, S, quantum_instance, H):
    # value: expectation value <psi|H|psi> of given Hamlitonian H
    qc, param_def_dict = build_circuit(qc_data, num_qubits, S)

    # build measurement circuits
    grouper = AbelianGrouper()
    groups = grouper.convert(H)
    circuits = []
    for group in groups:
        basis = ['I'] * group.num_qubits
        for pauli_string in group.primitive.paulis:
            for i, pauli in enumerate(pauli_string):
                p = str(pauli)
                if p != 'I':
                    if basis[i] == 'I':
                        basis[i] = p
                    elif basis[i] != p:
                        raise ValueError('PauliSumOp contains non-commuting terms!')
        new_qc = qc.copy()
        for i, pauli in enumerate(basis):
            if pauli == 'X':  # H @ X @ H = Z
                new_qc.h(i)
            if pauli == 'Y':  # S^dag @ H @ Y @ H @ S = Z
                new_qc.s(i)
                new_qc.h(i)
        circuits.append(new_qc)

    # check simulator
    sv_sim = quantum_instance.is_statevector

    # traverse measurement circuits
    value = 0
    for group, circuit in zip(groups, circuits):
        if sv_sim:
            sv = evaluate_circuit(qc, param_def_dict, quantum_instance, counts=False, sv=True, add_measurement=False)
            probabilities = sv.probabilities_dict()
        else:
            counts = evaluate_circuit(qc, param_def_dict, quantum_instance, counts=True, sv=False, add_measurement=True)
            shots = sum(counts.values())
            probabilities = {b: c / shots for b, c in counts.items()}
        for (pauli, coeff) in zip(group.primitive.paulis, group.primitive.coeffs):
            val = 0
            p = str(pauli)
            for b, prob in probabilities.items():
                val += prob * np.prod([(-1) ** (b[k] == '1' and p[k] != 'I') for k in range(len(b))])
            value += np.real(coeff * val)
    return value


def value_Expr(qc_data, num_qubits, S, quantum_instance, rng, num_samples, bins, p_lim_fun):
    # value: expressibility of parameterized circuit based on num_samples samples (following arXiv:1905.10876v1)
    qc, param_def_dict = build_circuit(qc_data, num_qubits, S)
    if p_lim_fun is None:
        p_lim_fun = lambda p: (-2 * np.pi, 2 * np.pi)

    # expressibility tools
    def statevector_overlap(sv1, sv2):
        F = np.abs(sv1.inner(sv2)) ** 2
        F = np.clip(F, 0, 1)
        return F

    def calculate_Fhaar_distribution(num_qubits, bin_edges):
        N = 2 ** num_qubits
        F_hist_haar = []
        for idx in range(len(bin_edges) - 1):
            a = bin_edges[idx]
            b = bin_edges[idx + 1]
            p = (1 - a) ** (N - 1) - (1 - b) ** (N - 1)  # integral over P = (N-1)*(1-F)**(N-2) from a to b
            F_hist_haar.append(p)
        return np.array(F_hist_haar)

    def calculate_F_distribution(F_list, bins):
        F_hist, F_bin_edges = np.histogram(F_list, bins=bins, range=(0, 1))
        F_hist = F_hist / len(F_list)
        return F_hist, F_bin_edges

    def kl_div(hist1, hist2, epsilon=1e-14):
        hist1[hist1 <= epsilon] = epsilon
        hist2[hist2 <= epsilon] = epsilon
        return entropy(hist1, hist2)

    def estimate_expressibility(F_hist, F_hist_haar):
        return kl_div(F_hist, F_hist_haar)

    # calculate expressibility
    if len(param_def_dict) > 0:
        F_list = np.empty(num_samples)
        for idx in range(num_samples):
            param_def_dict_list = [{p: rng.uniform(p_lim_fun(p)[0], p_lim_fun(p)[1]) for p in param_def_dict.keys()} for
                                   _ in range(2)]
            sv1, sv2 = evaluate_circuits([qc for _ in range(2)], param_def_dict_list, quantum_instance, counts=False,
                                         sv=True, add_measurement=False)
            F = statevector_overlap(sv1, sv2)
            F_list[idx] = F
    else:
        F_list = np.ones(num_samples)
    F_hist, F_bin_edges = calculate_F_distribution(F_list, bins)
    F_hist_haar = calculate_Fhaar_distribution(num_qubits, F_bin_edges)
    Expr = estimate_expressibility(F_hist, F_hist_haar)

    # return expressibility as value
    value = Expr
    return value


def value_Ent(qc_data, num_qubits, S, quantum_instance, rng, num_samples, p_lim_fun, eps=1e-8):
    # value: entanglement capability of parameterized circuit based on num_samples samples (following arXiv:1905.10876v1)
    qc, param_def_dict = build_circuit(qc_data, num_qubits, S)
    if p_lim_fun is None:
        p_lim_fun = lambda p: (-2 * np.pi, 2 * np.pi)

    # entanglement capability tools
    def get_reduced_sv(sv, j, b, eps=eps):
        num_qubits = int(np.log2(len(sv)))
        reduced_sv = np.zeros(2 ** (num_qubits - 1), dtype=np.csingle)
        for idx, key in enumerate(product('01', repeat=num_qubits)):
            key = ''.join(key)
            if np.abs(sv[idx]) <= eps:
                continue
            reduced_key = key[:j] + key[j + 1:]
            reduced_idx = int(np.argmax(Statevector.from_label(reduced_key)))
            index_key_int = int(key[j])
            reduced_sv[reduced_idx] += sv[idx] * (index_key_int == b)
        return reduced_sv  # not normalized?

    def get_D(u, v):
        M = np.outer(u, v)
        return np.sum(np.abs(M - M.transpose()) ** 2) / 2

    def get_Q(sv):
        sv = np.asarray(sv).astype(np.csingle)
        sv /= np.sqrt(np.sum(np.abs(sv) ** 2))  # ensure normalization
        num_qubits = int(np.log2(len(sv)))
        q = 0
        for j in range(num_qubits):
            reduced_sv_0 = get_reduced_sv(sv, j, 0)
            reduced_sv_1 = get_reduced_sv(sv, j, 1)
            d = get_D(reduced_sv_0, reduced_sv_1)
            q += d
        return q * 4 / num_qubits

    def estimate_entanglement(Q_list):
        Q_mean = np.mean(Q_list)
        return Q_mean

    # calculate entanglement capability
    if len(param_def_dict) == 0:
        num_samples = 1
    Q_list = np.empty(num_samples)
    for idx in range(num_samples):
        param_def_dict_rand = {p: rng.uniform(p_lim_fun(p)[0], p_lim_fun(p)[1]) for p in param_def_dict.keys()}
        sv = evaluate_circuit(qc, param_def_dict_rand, quantum_instance, counts=False, sv=True, add_measurement=False)
        Q = get_Q(sv)
        Q_list[idx] = Q
    Ent = estimate_entanglement(Q_list)

    # return entanglement capability as value
    value = Ent
    return value


def value_bits_fun(qc_data, num_qubits, S, quantum_instance, bits_fun):
    # value: extraction of measured bits with a custom function
    qc, param_def_dict = build_circuit(qc_data, num_qubits, S)
    counts = evaluate_circuit(qc, param_def_dict, quantum_instance, counts=True, sv=False)
    shots = sum(counts.values())
    value = 0
    for bits, count in counts.items():
        bits = [int(i) for i in bits[::-1]]
        value += bits_fun(bits) * count
    value /= shots
    return value
