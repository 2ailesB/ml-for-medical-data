def format_paramsdict(params_dict):
    temp = []
    for params in params_dict:
        pstr = ''
        for k, p in params.items():
            pstr += k[7:] + ' : '+ str(p) + '\n'
        temp.append(pstr)

    return temp