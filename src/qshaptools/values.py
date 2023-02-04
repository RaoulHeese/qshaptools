def value_dummy(S, const=1, **kwargs):
    return const


def value_fun_batch_wrapper_base(S_list, wrapped_value_fun, **kwargs):
    # for shapley_iteration_batch
    value_list = []
    for S in S_list:
        value_list.append(wrapped_value_fun(S, **kwargs))
    return value_list
