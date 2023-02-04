import numpy as np
from tqdm import tqdm

from tools import p_coalition, powerset


def w_calculation(S, F):
    return p_coalition(len(S), len(F))


def d_calculation(v, vi, delta_exponent=1):
    d = (vi - v) ** delta_exponent
    return d


def delta_phi_calculation(S, F, v, vi, use_weight=True, delta_exponent=1):
    if use_weight:
        w = w_calculation(S, F)
    else:
        w = 1
    d = d_calculation(v, vi, delta_exponent)
    delta_phi = w * d
    return delta_phi


class ShapleyValues():
    def __init__(self, unlocked_instructions, locked_instructions, value_fun, value_kwargs_dict, shap_sample_frac,
                 shap_sample_reps, shap_batch_size, evaluate_value_only_once, shap_sample_seed, memory, callback,
                 delta_exponent, name, silent):
        # setup instructions
        if locked_instructions is None:
            locked_instructions = []
        self._locked_instructions = sorted(list(locked_instructions))
        unlocked_instructions = list(unlocked_instructions)
        if len(unlocked_instructions) == 0:
            raise NotImplementedError
        self._unlocked_instructions = list(unlocked_instructions)

        # setup shap_sample_frac
        if shap_sample_frac is not None:
            if shap_sample_frac < 0:
                P_length = 2 ** (len(unlocked_instructions) - 1)
                shap_sample_frac = abs(int(shap_sample_frac)) / P_length
            elif shap_sample_frac == 0:
                raise NotImplementedError
        self._shap_sample_frac = shap_sample_frac

        # setup shap_sample_reps
        self._evaluate_value_only_once = bool(evaluate_value_only_once)
        if shap_sample_reps is None:
            shap_sample_reps = 1
        self._shap_sample_reps = shap_sample_reps

        # setup memory
        if memory is None:
            memory = dict()
        self._memory = memory

        # setup title
        if name is not None and len(name) == 0:
            name = None
        self._name = name

        # setup other properties
        self._shap_sample_seed = shap_sample_seed
        self._value_fun = value_fun
        self._value_kwargs_dict = value_kwargs_dict
        self._shap_batch_size = shap_batch_size
        self._callback = callback
        self._delta_exponent = delta_exponent
        self._silent = bool(silent)

    @property
    def phi_dict(self):
        if hasattr(self, 'phi_dict_'):
            return self.phi_dict_
        return None

    @property
    def memory(self):
        return self._memory

    @property
    def name(self):
        if self._name is not None:
            return f'{self._name}'
        else:
            return 'shap'

    def _evaluate_value_fun(self, S):
        if self._callback is not None:
            self._callback(S)
        if self._shap_batch_size is None:
            return self._value_fun(S=S, **self._value_kwargs_dict)
        else:
            return self._value_fun(S_list=S, **self._value_kwargs_dict)

    def _build_S_gen(self):
        F = self._unlocked_instructions
        if self._shap_sample_frac is not None:
            # subsample coalition S ~ w for every player
            total = len(F)
            with tqdm(desc=f'{self.name}:samp', total=total, disable=self._silent) as prog:
                self.S_gen_ = dict()
                self.S_gen_length_ = 0
                self.num_samples_dict_ = dict()
                for idx in F:
                    Fi = F.copy()
                    Fi.pop(F.index(idx))
                    P, P_length = powerset(Fi)
                    num_samples = int(np.ceil(self._shap_sample_frac * P_length))
                    i_array = np.arange(P_length)
                    P_array = [S for S in P]
                    p_array = [p_coalition(len(S), len(F)) for S in P_array]
                    i_samples = self.rng_.choice(i_array, size=num_samples, replace=True, p=p_array)
                    S_list = [P_array[i] for i in i_samples]
                    self.S_gen_[idx] = S_list
                    self.S_gen_length_ += num_samples
                    self.num_samples_dict_[idx] = num_samples
                    prog.update(1)
        else:
            # use all 2^N coalitions
            self.S_gen_, self.S_gen_length_ = powerset(F)
            self.num_samples_dict_ = None

    def _build_Si_total_list(self):
        self.Si_total_list_ = []
        total = self.S_gen_length_
        with tqdm(desc=f'{self.name}:jobs', total=total, disable=self._silent) as prog:
            if self._shap_sample_frac is not None:
                for idx, S_list in self.S_gen_.items():
                    for S in S_list:
                        S = sorted(list(S))
                        Si = sorted(S + [idx])
                        for _ in range(self._shap_sample_reps):
                            self.Si_total_list_.append([idx, S])
                        for _ in range(self._shap_sample_reps):
                            self.Si_total_list_.append([idx, Si])
                        prog.update(1)
            else:
                S_list = self.S_gen_
                for S in S_list:
                    S = sorted(list(S))
                    for _ in range(self._shap_sample_reps):
                        self.Si_total_list_.append([None, S])
                    prog.update(1)

    def _eval_Si_total_list(self):
        L = self._locked_instructions
        if self._evaluate_value_only_once:
            Si_effective_total_list = list(
                list(x) for x in set(tuple(x) for x in self.Si_total_list_))  # remove duplicates
        else:
            Si_effective_total_list = self.Si_total_list_
        if self._shap_batch_size is not None:
            Si_effective_total_list = [Si_batch_list.tolist() for Si_batch_list in
                                       np.array_split(Si_effective_total_list,
                                                      np.ceil(len(Si_effective_total_list) / self._shap_batch_size))]
        total = sum([len(Si_batch_list) for Si_batch_list in
                     Si_effective_total_list]) if self._shap_batch_size is not None else len(Si_effective_total_list)
        with tqdm(desc=f'{self.name}:eval', total=total, disable=self._silent) as prog:
            if self._shap_batch_size is not None:
                for Si_batch_list in Si_effective_total_list:
                    S_batch_list = [sorted(Si[1] + L) for Si in Si_batch_list]
                    value_list = self._evaluate_value_fun(S_batch_list)
                    for Si, value in zip(Si_batch_list, value_list):
                        i = Si[0]
                        key = tuple(sorted(Si[1]))
                        if key not in self._memory:
                            self._memory[key] = []
                        self._memory[key].append([i, value])
                        prog.update(1)
            else:
                for Si in Si_effective_total_list:
                    i = Si[0]
                    S = sorted(Si[1])
                    Sl = sorted(S + L)
                    value = self._evaluate_value_fun(Sl)
                    key = tuple(S)
                    if key not in self._memory:
                        self._memory[key] = []
                    self._memory[key].append([i, value])
                    prog.update(1)

    def _eval_shap_idx(self, idx, prog):
        F = self._unlocked_instructions
        if self._shap_sample_frac is not None:
            P = self.S_gen_[idx]
            use_weight = False
        else:
            Fi = F.copy()
            Fi.pop(F.index(idx))
            P, _ = powerset(Fi)
            use_weight = True
        phi = 0
        for S in P:
            S = list(sorted(S))
            key_S = tuple(S)
            key_Si = tuple(sorted(S + [idx]))
            assert key_S in self._memory and key_Si in self._memory  # sanity check, should never be violated
            v = np.mean([x[1] for x in self._memory[key_S]])
            vi = np.mean([x[1] for x in self._memory[key_Si]])
            delta_phi = delta_phi_calculation(S, F, v, vi, use_weight=use_weight, delta_exponent=self._delta_exponent)
            phi += delta_phi
            prog.update(1)
        if self._shap_sample_frac is not None:
            phi /= len(P)
        return phi

    def _eval_shap(self):
        F = self._unlocked_instructions
        self.phi_dict_ = {}
        N = len(F)
        if self._shap_sample_frac is not None:
            self._n_total_effective = sum([len(S_list) for S_list in self.S_gen_.values()])
        else:
            self._n_total_effective = N * 2 ** (N - 1)
        with tqdm(desc=f'{self.name}:sums', total=self._n_total_effective, disable=self._silent) as prog:
            for idx in F:
                phi = self._eval_shap_idx(idx, prog)
                self.phi_dict_[idx] = phi

    def __call__(self):
        self.rng_ = np.random.RandomState(self._shap_sample_seed)
        self._build_S_gen()
        self._build_Si_total_list()
        self._eval_Si_total_list()
        self._eval_shap()
        return self.phi_dict

    def __str__(self):
        # print settings
        N = len(self._unlocked_instructions)
        M = len(self._locked_instructions)
        if self._shap_sample_frac is not None:
            self._n_per_phi = int(np.ceil(self._shap_sample_frac * 2 ** (N - 1)))  # each evaluated twice: S and S+i
        else:
            self._n_per_phi = 2 ** (N - 1)
        self._n_total = N * self._n_per_phi
        self._n_valfun = 2 ** N
        rep = f'[{self.name}]\n'
        rep += f'value_fun:                   {str(self._value_fun)}\n'
        rep += f'unlocked_instructions [{N:3d}]: {self._unlocked_instructions}\n'
        rep += f'locked_instructions   [{M:3d}]: {self._locked_instructions}\n'
        rep += f'delta_exponent:              {self._delta_exponent}\n'
        rep += f'shap_sample_frac:            {self._shap_sample_frac}\n'
        rep += f'shap_sample_reps:            {self._shap_sample_reps}\n'
        rep += f'evaluate_value_only_once:    {self._evaluate_value_only_once}\n'
        rep += f'shap_sample_seed:            {self._shap_sample_seed}\n'
        rep += f'shap_batch_size:             {self._shap_batch_size}\n'
        rep += f'possible value functions:    {self._n_valfun}\n'
        rep += f'terms per phi:               {self._n_per_phi}\n'
        rep += f'total shapley terms:         {self._n_total}'
        return rep

    def clear_memory(self):
        self._memory = dict()

    def eval_S_list(self, S_list, recall):
        L = self._locked_instructions
        S_list = [list(S) for S in S_list]
        values = []
        total = len(S_list)
        with tqdm(desc=f'{self.name}:vals', total=total, disable=self._silent) as prog:
            for S in S_list:
                S = sorted(list(S))
                Sl = sorted(S + L)
                key = tuple(S)
                if recall and key in self._memory:
                    value = np.mean([vi[1] for vi in self._memory[key]])
                else:
                    if self._shap_batch_size is None:
                        value = self._evaluate_value_fun(Sl)
                    else:
                        value = self._evaluate_value_fun([Sl])[0]
                values.append(value)
                prog.update(1)
        return values

    def get_summary_dict(self, property_list=None):
        if property_list is None:
            property_list = []

        def get_attr(name_):
            return getattr(self, name_) if hasattr(self, name_) else None

        summary = {'name': self.name,
                   'value_fun': get_attr('_value_fun'),
                   'unlocked_instructions': get_attr('_unlocked_instructions'),
                   'locked_instructions': get_attr('_locked_instructions'),
                   'delta_exponent': get_attr('_delta_exponent'),
                   'shap_sample_frac': get_attr('_shap_sample_frac'),
                   'shap_sample_reps': get_attr('_shap_sample_reps'),
                   'evaluate_value_only_once': get_attr('_evaluate_value_only_once'),
                   'shap_sample_seed': get_attr('_shap_sample_seed'),
                   'shap_batch_size': get_attr('_shap_batch_size'),
                   'n_valfun': get_attr('_n_valfun'),
                   'n_per_phi': get_attr('_n_per_phi'),
                   'n_total': get_attr('_n_total'),
                   'n_total_effective': get_attr('_n_total_effective'),
                   'S_gen_length': get_attr('S_gen_length_'),
                   'num_samples_dict': get_attr('num_samples_dict_'),
                   'phi_dict': get_attr('phi_dict_')
                   }
        for name in property_list:
            summary.update({name: get_attr(name)})
        return summary
