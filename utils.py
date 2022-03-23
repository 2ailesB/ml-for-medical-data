def format_paramsdict(params_dict):
    for p in params_dict:
        for v in p.items():
            p[v[0]] = '\n ' + str(v[1])

    return params_dict