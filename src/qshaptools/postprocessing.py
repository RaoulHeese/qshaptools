import numpy as np
from tqdm import tqdm
from tools import powerset
from ushap import delta_phi_calculation, w_calculation, d_calculation


def shapley_value_from_memory_extended(unlocked_instructions, memory, K,
                                       memory_val_fun=lambda memory, K, key, i: np.mean([x[1] for x in memory[key]]),
                                       memory_count_fun=lambda memory, K, key, i: np.sum(
                                           [1 for x in memory[key] if x[0] == i or x[0] is None]),
                                       delta_exponent=1, desc='shap_val'):
    total = len(unlocked_instructions) * 2 ** (len(unlocked_instructions) - 1)
    phi_dict = dict()
    with tqdm(desc=desc, total=total) as prog:
        for i in unlocked_instructions:
            F = unlocked_instructions
            Fi = F.copy()
            Fi.pop(F.index(i))
            P, _ = powerset(Fi)
            if K is not None:
                n = 0
            else:
                n = 1
            phi = 0
            for k, S in enumerate(P):
                S = sorted(list(S))
                Si = sorted(S + [i])
                key_v = tuple(S)
                key_vi = tuple(Si)
                if key_v in memory and key_vi in memory:
                    if K is not None:
                        count_i = memory_count_fun(memory, K, key_v, i)
                        count_ii = memory_count_fun(memory, K, key_vi, i)
                        assert count_i == count_ii
                        #
                        if count_i > 0 and count_ii > 0:
                            v = np.mean(memory_val_fun(memory, K, key_v, i))
                            vi = np.mean(memory_val_fun(memory, K, key_vi, i))
                            #
                            n_count = count_i // K
                            assert float(n_count) == count_i / K
                            n += n_count
                            #
                            phi += n_count * delta_phi_calculation(S, F, v, vi, use_weight=False,
                                                                   delta_exponent=delta_exponent)
                    else:
                        v = memory_val_fun(memory, K, key_v, i)
                        vi = memory_val_fun(memory, K, key_vi, i)
                        phi += delta_phi_calculation(S, F, v, vi, use_weight=True, delta_exponent=delta_exponent)
                prog.update(1)
            phi /= n
            phi_dict[i] = phi
    return phi_dict


def shapley_value_from_memory(unlocked_instructions, memory, K):
    return shapley_value_from_memory_extended(unlocked_instructions, memory, K)


def shapley_p_from_memory(all_instructions, locked_instructions, memory, verify=False, verify_epsilon=1e-9,
                          include_locked_instructions_in_key=True):
    assert all((idx in all_instructions for idx in locked_instructions))
    assert type(memory) is dict
    unlocked_instructions = [idx for idx in all_instructions if idx not in locked_instructions]
    total = len(unlocked_instructions) * 2 ** (len(unlocked_instructions) - 1)
    #
    p_dict = dict()
    if verify:
        w_sum = dict()
    with tqdm(desc='shap_p', total=total) as prog:
        for i in unlocked_instructions:
            p_dict[i] = {}
            F = unlocked_instructions
            Fi = F.copy()
            Fi.pop(F.index(i))
            P, P_length = powerset(Fi)
            for k, S in enumerate(P):
                S = list(S)
                Si = S + [i]
                Sl = sorted(S + locked_instructions if include_locked_instructions_in_key else [])
                Sil = sorted(Si + locked_instructions if include_locked_instructions_in_key else [])
                key_v = tuple(Sl)
                key_vi = tuple(Sil)
                if key_v in memory and key_vi in memory:
                    v = np.mean([vi[1] for vi in memory[key_v]])
                    vi = np.mean([vi[1] for vi in memory[key_vi]])
                    w = w_calculation(S, F)
                    d = d_calculation(v, vi, delta_exponent=1)
                    if d not in p_dict[i]:
                        p_dict[i][d] = 0
                    p_dict[i][d] += w
                    if verify:
                        if i not in w_sum:
                            w_sum[i] = 0
                        w_sum[i] += w
                prog.update(1)
    if verify:
        assert all([np.abs(w - 1) <= np.abs(verify_epsilon) for w in w_sum.values()]), f'{w_sum}'
    return p_dict
