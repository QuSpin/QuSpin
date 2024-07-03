import warnings


def check_T(sort_opstr, operator_list, L, a):
    missing_ops = []
    for i in range(0, L // a, 1):
        for op in operator_list:
            opstr = str(op[0])
            indx = list(op[1])
            for j, ind in enumerate(indx):
                indx[j] = (ind + i * a) % L

            new_op = list(op)
            new_op[1] = indx

            new_op = sort_opstr(new_op)

            if not (new_op in operator_list):
                missing_ops.append(new_op)

    return missing_ops


def check_Z(sort_opstr, operator_list, photon):
    missing_ops = []

    for op in operator_list:
        opstr = str(op[0])
        indx = list(op[1])

        if photon:
            opstr_left, opstr_right, opstr_phtn = opstr.split("|")
            N_left = len(opstr_left.replace("n", "").replace("z", ""))
            N_right = len(opstr_right.replace("n", "").replace("z", ""))
            new_opstr = "|".join((opstr_right, opstr_left, opstr_phtn))
        else:
            opstr_left, opstr_right = opstr.split("|")
            N_left = len(opstr_left.replace("n", "").replace("z", ""))
            N_right = len(opstr_right.replace("n", "").replace("z", ""))
            new_opstr = "|".join((opstr_right, opstr_left))

        new_op = list(op)
        new_op[0] = new_opstr
        new_op[2] *= (-1) ** (N_right * N_left)
        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return missing_ops


def check_P(sort_opstr, operator_list, L):
    missing_ops = []
    for op in operator_list:
        indx = list(op[1])
        for j, ind in enumerate(indx):
            indx[j] = (L - 1 - ind) % L

        new_op = list(op)
        new_op[1] = indx
        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return missing_ops


def check_PZ(sort_opstr, operator_list, L, photon):
    missing_ops = []
    for op in operator_list:
        opstr = str(op[0])
        indx = list(op[1])

        if photon:
            opstr_left, opstr_right, opstr_phtn = opstr.split("|")
            N_left = len(opstr_left.replace("n", "").replace("z", ""))
            N_right = len(opstr_right.replace("n", "").replace("z", ""))
            new_opstr = "|".join((opstr_right, opstr_left, opstr_phtn))
        else:
            opstr_left, opstr_right = opstr.split("|")
            N_left = len(opstr_left.replace("n", "").replace("z", ""))
            N_right = len(opstr_right.replace("n", "").replace("z", ""))
            new_opstr = "|".join((opstr_right, opstr_left))

        new_op = list(op)
        new_op[0] = new_opstr
        new_op[1] = indx
        new_op[2] *= (-1) ** (N_right * N_left)
        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return missing_ops
