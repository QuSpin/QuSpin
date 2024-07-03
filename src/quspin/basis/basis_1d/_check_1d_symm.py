import warnings


def flip_sublat(opstr, indx, lat=0):
    sign = 1
    opstr = [str(s) for s in opstr]
    for s, i, j in zip(opstr, indx, range(len(indx))):
        if (i % 2) == (lat % 2):
            if s in ["z", "y"]:
                sign *= -1
            elif s == "+":
                opstr[j] = "-"
            elif s == "-":
                opstr[j] = "+"

    return sign, "".join(opstr)


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


def check_Z(sort_opstr, operator_list):
    missing_ops = []
    odd_ops = []
    for op in operator_list:
        opstr = str(op[0])
        indx = list(op[1])

        if opstr.count("|") == 1:
            i = opstr.index("|")
        else:
            i = len(opstr)

        z_count = opstr[:i].count("z")
        y_count = opstr[:i].count("y")
        if ((y_count + z_count) % 2) != 0:
            odd_ops.append(op)

        new_op = list(op)
        new_op[0] = (
            new_op[0][:i].replace("+", "#").replace("-", "+").replace("#", "-")
            + op[0][i:]
        )
        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return odd_ops, missing_ops


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


def check_PZ(sort_opstr, operator_list, L):
    missing_ops = []
    for op in operator_list:
        opstr = str(op[0])
        indx = list(op[1])

        if opstr.count("|") == 1:
            i = opstr.index("|")
        else:
            i = len(opstr)

        for j, ind in enumerate(indx):
            indx[j] = (L - 1 - ind) % L

        sign = (-1) ** (opstr[:i].count("z") + opstr.count("y"))

        new_op = list(op)
        new_op[0] = (
            new_op[0][:i].replace("+", "#").replace("-", "+").replace("#", "-")
            + op[0][i:]
        )
        new_op[1] = indx
        new_op[2] *= sign
        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return missing_ops


def check_ZA(sort_opstr, operator_list):
    missing_ops = []
    odd_ops = []

    for op in operator_list:

        opstr = str(op[0])
        indx = list(op[1])

        if opstr.count("|") == 1:
            i = opstr.index("|")
        else:
            i = len(opstr)

        sign, new_opstr = flip_sublat(opstr[:i], indx[:i], lat=0)

        if sign == -1:
            odd_ops.append(op)

        new_op = list(op)
        new_op[0] = new_opstr + opstr[i:]
        new_op = sort_opstr(new_op)

        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return odd_ops, missing_ops


def check_ZB(sort_opstr, operator_list):
    missing_ops = []
    odd_ops = []

    for op in operator_list:

        opstr = str(op[0])
        indx = list(op[1])

        if opstr.count("|") == 1:
            i = opstr.index("|")
        else:
            i = len(opstr)

        sign, new_opstr = flip_sublat(opstr[:i], indx[:i], lat=1)

        if sign == -1:
            odd_ops.append(op)

        new_op = list(op)
        new_op[0] = new_opstr + opstr[i:]
        new_op = sort_opstr(new_op)

        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return odd_ops, missing_ops
