

cdef int op_template(basis_type[:] op_pars, npy_intp Ns, basis_type[:] basis,
                     str opstr, NP_INT32_t *indx, scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):

    cdef npy_intp i
    cdef int error

    for i in range(Ns):
        col[i] = i

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    for i in range(Ns):
        row[i] = Ns - row[i] - 1

    return error




cdef int n_op_template(basis_type[:] op_pars, npy_intp Ns, basis_type[:] basis,
            str opstr, NP_INT32_t *indx, scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef npy_intp i
    cdef basis_type s
    cdef int error = 0
    cdef bool found

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        if row[i]!=basis[i]:
            row[i] = findzstate(basis,Ns,row[i],&found)
        else:
            row[i]=i
            found=True

        if not found:
            ME[i] = _np.nan


    return error








cdef int p_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int pblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):

    cdef basis_type s
    cdef npy_intp i
    cdef int error = 0
    cdef int q
    cdef double n
    cdef bool found
    cdef NP_INT8_t sign=1

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_P_template(row[i],L,&sign,&q,ref_pars)

        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True
        
        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= sign*(pblock**q)*sqrt(n)

    return error




cdef int pz_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int pzblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,qg
    cdef double n
    cdef bool found
    cdef NP_INT8_t sign=1

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_PZ_template(row[i],L,&sign,&qg,ref_pars)
        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]

        ME[i] *= sign*(pzblock**qg)*sqrt(n)

    return error











cdef int p_z_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int pblock,int zblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,q,g
    cdef int R[2]
    cdef double n
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_P_Z_template(row[i],L,&sign,R,ref_pars)

        q = R[0]
        g = R[1]
        
        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]

        ME[i] *= sign*sqrt(n)*(pblock**q)*(zblock**g)

    return error






















cdef int t_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int a, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,l
    cdef double n,k
    cdef bool found
    cdef NP_INT8_t sign=1

    k = (2.0*_np.pi*kblock*a)/L
    l = 0
    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if (matrix_type is float or matrix_type is double) and ((2*a*kblock) % L) != 0:
        error = -1

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_T_template(row[i],L,a,&sign,&l,ref_pars)

        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[i]
        n /= N[s]
        n = sqrt(n)
        ME[i] *= n

        if (matrix_type is float or matrix_type is double):
            ME[i] *= sign*(-1.0)**(l*2*a*kblock/L)
        else:
            ME[i] *= sign*(cos(k*l) - 1.0j * sin(k*l))

            

    return error






cdef double MatrixElement_T_P(int L,int pblock, int kblock, int a, int l, double k, int q,int Nc,int Nr,int mc,int mr):
    cdef double nc,nr
    cdef double ME
    cdef int sr,sc,fr,fc,mmr,mmc

    if Nc > 0:
        sc = 1
    else:
        sc = -1

    if Nr > 0:
        sr = 1
    else:
        sr = -1

    mmc = (mc % (L+1)) - 1
    mmr = (mr % (L+1)) - 1

    if mmc >= 0:
        fc = 2*(mc//(L+1)) - 1
        nc = (1 + fc*sc*pblock*cos(k*mmc))/Nc
    else:
        fc = 1
        nc = 1.0/Nc
    nc *= sc

    if mmr >= 0:
        fr = 2*(mr//(L+1)) - 1
        nr = (1 + fr*sr*pblock*cos(k*mmr))/Nr
    else:
        fr = 1
        nr = 1.0/Nr
    nr *= sr

    ME=sqrt(nr/nc)*(sc*pblock)**q

    if sr == sc :
        if mmr < 0:
            ME *= cos(k*l)
        else:
            ME *= (cos(k*l)+fr*sc*pblock*cos((l-mmr)*k))/(1+fr*sc*pblock*cos(k*mmr))
    else:
        if mmr < 0:
            ME *= -sc*sin(k*l)
        else:
            ME *= (-sc*sin(k*l)+fr*pblock*sin((l-mmr)*k))/(1-fr*sc*pblock*cos(k*mmr))        

    return ME






cdef int t_p_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int pblock,int a, npy_intp Ns,
                        N_type *N, M1_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,q
    cdef int R[2]
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0

    cdef double k = (2.0*_np.pi*kblock*a)/L

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
            ME[Ns+i] = _np.nan
            row[Ns+i] = i
            col[i] = i
            col[Ns+i] = i

    if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
        for i in range(Ns):
            sign = 1
            s = RefState_T_P_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            q = R[1]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= sign*MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


    else:
        for i in range(Ns):

            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            sign = 1
            s = RefState_T_P_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            q = R[1]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                for j in range(i,i+o,1):
                    ME[j] = _np.nan
                continue


            if (ss == i) and (q == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= sign*MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

            else: # off diagonal ME

                if (ss > 0) and (basis[ss] == basis[ss-1]):
                    ss -= 1; p = 2
                elif (ss < (Ns - 1)) and (basis[ss] == basis[ss+1]):
                    p = 2
                else:
                    p = 1

                for c in range(o):
                    me[c] = ME[i+c] 

                for c in range(o):
                    for b in range(p):
                        j = i + c + Ns*b
                        row[j] = ss + b
                        ME[j] = sign*me[c]*MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error





cdef double MatrixElement_T_PZ(int L,int pzblock, int kblock, int a, int l, double k, int q,int Nc,int Nr,int mc,int mr):
    cdef double nc,nr
    cdef double ME
    cdef int sr,sc,fr,fc,mmr,mmc

    if Nc > 0:
        sc = 1
    else:
        sc = -1

    if Nr > 0:
        sr = 1
    else:
        sr = -1

    mmc = (mc % (L+1)) - 1
    mmr = (mr % (L+1)) - 1



    if mmc >= 0:
        fc =  2*(mc//(L+1)) - 1
        nc = (1 + fc*sc*pzblock*cos(k*mmc))/Nc
    else:
        fc = 1
        nc = 1.0/Nc

    nc *= sc

    if mmr >= 0:
        fr =  2*(mr//(L+1)) - 1
        nr = (1 + fr*sr*pzblock*cos(k*mmr))/Nr
    else:
        fr = 1
        nr = 1.0/Nr
    nr *= sr

    ME=sqrt(nr/nc)*(sc*pzblock)**q

    if sr == sc :
        if mmr < 0:
            ME *= cos(k*l)
        else:
            ME *= (cos(k*l)+fr*sc*pzblock*cos((l-mmr)*k))/(1+fr*sc*pzblock*cos(k*mmr))
    else:
        if mmr < 0:
            ME *= -sc*sin(k*l)
        else:
            ME *= (-sc*sin(k*l)+fr*pzblock*sin((l-mmr)*k))/(1-fr*sc*pzblock*cos(k*mmr))        

    return ME




cdef int t_pz_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int pzblock,int a,
                        npy_intp Ns, N_type *N, M1_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx,
                        scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,qg
    cdef int R[2]
    cdef bool found
    cdef NP_INT8_t sign=1

    cdef double k = (2.0*_np.pi*kblock*a)/L

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        ME[Ns+i] = _np.nan
        row[Ns+i] = i
        col[i] = i
        col[Ns+i] = i

    if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
        for i in range(Ns):
            sign = 1
            s = RefState_T_PZ_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            qg = R[1]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= sign*MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


    else:
        for i in range(Ns):
            sign = 1
            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            s = RefState_T_PZ_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            qg = R[1]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                for j in range(i,i+o,1):
                    ME[j] = _np.nan
                continue


            if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= sign*MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

            else: # off diagonal ME

                if (ss > 0) and (basis[ss] == basis[ss-1]):
                    ss -= 1; p = 2
                elif (ss < (Ns - 1)) and (basis[ss] == basis[ss+1]):
                    p = 2
                else:
                    p = 1

                for c in range(o):
                    me[c] = ME[i+c]

                for c in range(o):
                    for b in range(p):
                        j = i + c + Ns*b
                        row[j] = ss + b
                        ME[j] = sign*me[c]*MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error












cdef double MatrixElement_T_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, double k, int q, int g,int Nc,int Nr,int mc,int mr):
    cdef double nr,nc
    cdef double ME
    cdef int sr,sc
    cdef int nnr,mmr,cr,fr1,fr2,nnc,mmc,cc,fc1,fc2

    # define sign function
    if Nr > 0:
        sr = 1
    else:
        sr = -1

    if Nc > 0:
        sc = 1
    else:
        sc = -1

    # unpack long integer, cf Anders' notes
    mmc = mc % (L+1)
    mc /= (L+1)
    nnc = mc % (L+1)
    mc /= (L+1)

    mmr = mr % (L+1)
    mr /= (L+1)
    nnr = mr % (L+1)
    mr /= (L+1)


    nr = 1.0
    nc = 1.0
    cr = 1
    cc = 1
    fr1 = 1
    fr2 = 1
    fc1 = 1
    fc2 = 1

    if mr == 0:
        nr = 1.0/Nr
    elif mr == 1 or mr == 2:
        cr = 2
        fr1 = (-1 if mr==1 else 1)
        nr = (1.0 + fr1*pblock*sr*cos(k*mmr))/Nr
    elif mr == 3 or mr == 4:
        cr = 3
        fr1 = (-1 if mr==3 else 1)
        nr = (1.0 + fr1*zblock*cos(k*nnr))/Nr
    elif mr == 5 or mr == 6:
        cr = 4
        fr1 = (-1 if mr==5 else 1)
        nr = (1.0 + fr1*pblock*zblock*sr*cos(k*mmr))/Nr    
    elif mr == 7 or mr == 8:
        cr = 5
        fr1 = (-1 if mr==7 else 1)
        fr2 = -1
        nr = (1.0 + fr1*pblock*sr*cos(k*mmr))*(1.0 + fr2*zblock*cos(k*nnr))/Nr
    elif mr == 9 or mr == 10:
        cr = 5
        fr1 = (-1 if mr==9 else 1)
        nr = (1.0 + fr1*pblock*sr*cos(k*mmr))*(1.0 + fr2*zblock*cos(k*nnr))/Nr
    nr *= sr

    if mc == 0:
        nc = 1.0/Nc
    elif mc == 1 or mc == 2:
        cc = 2
        fc1 = (-1 if mc==1 else 1)
        nc = (1.0 + fc1*pblock*sc*cos(k*mmc))/Nc
    elif mc == 3 or mc == 4:
        cc = 3
        fc1 = (-1 if mc==3 else 1)
        nc = (1.0 + fc1*zblock*cos(k*nnc))/Nc
    elif mc == 5 or mc == 6:
        cc = 4
        fc1 = (-1 if mc==5 else 1)
        nc = (1.0 + fc1*pblock*zblock*sc*cos(k*mmc))/Nc    
    elif mc == 7 or mc == 8:
        cc = 5
        fc1 = (-1 if mc==7 else 1)
        fc2 = -1
        nc = (1.0 + fc1*pblock*sc*cos(k*mmc))*(1.0 + fc2*zblock*cos(k*nnc))/Nc
    elif mc == 9 or mc == 10:
        cc = 5
        fc1 = (-1 if mc==9 else 1)
        nc = (1.0 + fc1*pblock*sc*cos(k*mmc))*(1.0 + fc2*zblock*cos(k*nnc))/Nc


    nc *= sc



    ME=sqrt(nr/nc)*((sc*pblock)**q)*(zblock**g)

    if sr == sc :
        if (cr == 1) or (cr == 3):
            ME *= cos(k*l)
        elif (cr == 2) or (cr == 5):
            ME *= (cos(k*l)+fr1*sc*pblock*cos((l-mmr)*k))/(1+fr1*sc*pblock*cos(k*mmr))
        elif (cr == 4):
            ME *= (cos(k*l)+fr1*sc*pblock*zblock*cos((l-mmr)*k))/(1+fr1*sc*pblock*zblock*cos(k*mmr))
    else:
        if (cr == 1) or (cr == 3):
            ME *= -sc*sin(k*l)
        elif (cr == 2) or (cr == 5):
            ME *= (-sc*sin(k*l) + fr1*pblock*sin((l-mmr)*k))/(1-fr1*sc*pblock*cos(k*mmr))
        elif (cr == 4):
            ME *= (-sc*sin(k*l) + fr1*pblock*zblock*sin((l-mmr)*k))/(1-fr1*sc*pblock*zblock*cos(k*mmr))

    return ME






cdef int t_p_z_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int pblock,int zblock,int a,
                        npy_intp Ns, N_type *N, M2_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx,
                        scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,q,g
    cdef int R[3]
    cdef bool found
    cdef NP_INT8_t sign=1

    cdef double k = (2.0*_np.pi*kblock*a)/L

    R[0] = 0
    R[1] = 0
    R[2] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        ME[Ns+i] = _np.nan
        row[Ns+i] = i
        col[i] = i
        col[Ns+i] = i

    if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
        for i in range(Ns):
            sign = 1
            s = RefState_T_P_Z_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            q = R[1]
            g = R[2]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= sign*MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])

    else:
        for i in range(Ns):
            sign = 1
            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            s = RefState_T_P_Z_template(row[i],L,a,&sign,R,ref_pars)

            l = R[0]
            q = R[1]
            g = R[2]

            if s!=basis[i]:
                ss = findzstate(basis,Ns,s,&found)
            else:
                ss = i
                found = True

            if not found:
                for j in range(i,i+o,1):
                    ME[j]  = _np.nan
                continue


            if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= sign*MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

            else: # off diagonal ME

                if (ss > 0) and (basis[ss] == basis[ss-1]):
                    ss -= 1; p = 2
                elif (ss < (Ns - 1)) and (basis[ss] == basis[ss+1]):
                    p = 2
                else:
                    p = 1

                for c in range(o):
                    me[c] = ME[i+c]

                for c in range(o):
                    for b in range(p):
                        j = i + c + Ns*b
                        row[j] = ss + b
                        ME[j] = sign*me[c]*MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error










cdef double complex MatrixElement_T_ZA(int L,int zAblock, int kblock, int a, int l, double k, int gA,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    
    mr -= 1
    mc -= 1

    if mr >= 0:
        nr = (1 + zAblock*cos(k*mr))/Nr
    else:
        nr = 1.0/Nr

    if mc >= 0:
        nc = (1 + zAblock*cos(k*mc))/Nc
    else:
        nc = 1.0/Nc


    ME=sqrt(nc/nr)*(zAblock**gA)

    if ((2*a*kblock) % L) == 0:
        ME *= (-1)**(2*l*a*kblock/L)
    else:
        ME *= (cos(k*l) - 1.0j * sin(k*l))

    return ME



cdef int t_zA_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int zAblock,int a, npy_intp Ns,
                        N_type *N, M1_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,gA
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if (matrix_type is float or matrix_type is double) and ((2*a*kblock) % L) != 0:
        error = -1

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_T_ZA_template(row[i],L,a,&sign,R,ref_pars)

        l = R[0]
        gA = R[1]

        if s!=basis[i]:
            ss = findzstate(basis,Ns,s,&found)
        else:
            ss = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue


        row[i] = ss


        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZA(L,zAblock,kblock,a,l,k,gA,N[i],N[ss],m[i],m[ss])
            ME[i] *= sign*me.real
        else:
            ME[i] *= sign*MatrixElement_T_ZA(L,zAblock,kblock,a,l,k,gA,N[i],N[ss],m[i],m[ss])

        
    return error














cdef double complex MatrixElement_T_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, double k, int gA, int gB,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    cdef int mmr,cr,mmc,cc


    mmc = mc % (L+1)
    cc = mc/(L+1)

    mmr = mr % (L+1)
    cr = mr/(L+1)

    nr = 1.0
    nc = 1.0


    if cr == 1:
        nr = 1.0/Nr
    elif cr == 2:
        nr = (1.0 + zAblock*cos(k*mmr) )/Nr
    elif cr == 3:
        nr = (1.0 + zBblock*cos(k*mmr) )/Nr
    elif cr == 4:
        nr = (1.0 + zAblock*zBblock*cos(k*mmr) )/Nr    
    

    if cc == 1:
        nc = 1.0/Nc
    elif cc == 2:
        nc = (1.0 + zAblock*cos(k*mmc) )/Nc
    elif cc == 3:
        nc = (1.0 + zBblock*cos(k*mmc) )/Nc
    elif cc == 4:
        nc = (1.0 + zAblock*zBblock*cos(k*mmc) )/Nc    
    

    ME=sqrt(nc/nr)*(zAblock**gA)*(zBblock**gB)


    if ((2*a*kblock) % L) == 0:
        ME *= (-1)**(2*l*a*kblock/L)
    else:
        ME *= (cos(k*l) - 1.0j * sin(k*l))

    return ME



cdef int t_zA_zB_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,
                        int L,int kblock,int zAblock,int zBblock,int a, npy_intp Ns, N_type *N, M2_type *m,
                        basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,gA,gB
    cdef int R[3]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0
    R[2] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if (matrix_type is float or matrix_type is double) and ((2*a*kblock) % L) != 0:
        error = -1

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_T_ZA_ZB_template(row[i],L,a,&sign,R,ref_pars)

        l = R[0]
        gA = R[1]
        gB = R[2]

        if s!=basis[i]:
            ss = findzstate(basis,Ns,s,&found)
        else:
            ss = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
            ME[i] *= sign*me.real
        else:
            ME[i] *= sign*MatrixElement_T_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])


    return error









cdef double complex MatrixElement_T_ZB(int L,int zBblock, int kblock, int a, int l, double k, int gB,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    
    mr -= 1
    mc -= 1

    if mr >=0:
        nr = (1 + zBblock*cos(k*mr))/Nr
    else:
        nr = 1.0/Nr

    if mc >= 0:
        nc = (1 + zBblock*cos(k*mc))/Nc
    else:
        nc = 1.0/Nc


    ME=sqrt(nc/nr)*(zBblock**gB)

    if (2*kblock*a % L) == 0:
        ME *= (-1)**(2*l*a*kblock/L)
    else:
        ME *= (cos(k*l) - 1.0j * sin(k*l))

    return ME



cdef int t_zB_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int zBblock,int a, npy_intp Ns,
                        N_type *N, M1_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i 
    cdef int error,l,gB
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if (matrix_type is float or matrix_type is double) and ((2*a*kblock) % L) != 0:
        error = -1

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_T_ZB_template(row[i],L,a,&sign,R,ref_pars)

        l = R[0]
        gB = R[1]

        if s!=basis[i]:
            ss = findzstate(basis,Ns,s,&found)
        else:
            ss = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])
            ME[i] *= sign*me.real
        else:
            ME[i] *= sign*MatrixElement_T_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])


    return error





cdef double complex MatrixElement_T_Z(int L,int zblock, int kblock, int a, int l, double k, int g,int Nc,int Nr,int mc,int mr):
    cdef double nr,nc
    cdef double complex ME
    cdef int sr,sc,mmr,mmc

    mmr = (mr % (L+1)) - 1
    sr = 2*(mr//(L+1))-1

    mmc = (mc % (L+1)) - 1
    sc = 2*(mc//(L+1))-1

    if mmc >= 0:
        nc = (1 + sc*zblock*cos(k*mmc))/Nc
    else:
        nc = 1.0/Nc

    if mmr >=0:
        nr = (1 + sr*zblock*cos(k*mmr))/Nr
    else:
        nr = 1.0/Nr

    ME=sqrt(nr/nc)*(zblock**g)


    if ((2*a*kblock) % L) == 0:
        ME *= (-1)**(2*l*a*kblock/L)
    else:
        ME *= (cos(k*l) - 1.0j * sin(k*l))

    # print ME,zblock,g,l
    return ME



cdef int t_z_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int kblock,int zblock,int a, npy_intp Ns,
                        N_type *N, M1_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,g
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found
    cdef NP_INT8_t sign=1

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if (matrix_type is float or matrix_type is double) and ((2*a*kblock) % L) != 0:
        error = -1

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_T_Z_template(row[i],L,a,&sign,R,ref_pars)

        l = R[0]
        g = R[1]

        if s!=basis[i]:
            ss = findzstate(basis,Ns,s,&found)
        else:
            ss = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_Z(L,zblock,kblock,a,l,k,g,N[i],N[ss],m[i],m[ss])
            ME[i] *= sign*me.real
        else:
            ME[i] *= sign*MatrixElement_T_Z(L,zblock,kblock,a,l,k,g,N[i],N[ss],m[i],m[ss])
            

    return error








cdef int zA_op_template( basis_type[:] op_pars,basis_type[:] ref_pars,int L,int zAblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,gA
    cdef bool found
    cdef NP_INT8_t sign
    cdef double n

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_ZA_template(row[i],L,&sign,&gA,ref_pars)

        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= sign*(zAblock)**gA*sqrt(n)


    return error







cdef int zA_zB_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,
                        int L,int zAblock,int zBblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,gA,gB
    cdef int R[2]
    cdef bool found
    cdef NP_INT8_t sign=1
    cdef double n

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_ZA_ZB_template(row[i],L,&sign,R,ref_pars)

        gA = R[0]
        gB = R[1]
        
        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= sign*(zAblock**gA)*(zBblock**gB)*sqrt(n)

    return error



cdef int zB_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int zBblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,gB
    cdef bool found
    cdef NP_INT8_t sign=1
    cdef double n

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_ZB_template(row[i],L,&sign,&gB,ref_pars)
        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= sign*(zBblock)**gB*sqrt(n)


    return error



cdef int z_op_template(basis_type[:] op_pars,basis_type[:] ref_pars,int L,int zblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,g
    cdef bool found
    cdef NP_INT8_t sign=1
    cdef double n


    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        sign = 1
        s = RefState_Z_template(row[i],L,&sign,&g,ref_pars)
        if s!=basis[i]:
            s = findzstate(basis,Ns,s,&found)
        else:
            s = i
            found = True

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= sign*(zblock if g else 1)*sqrt(n)


    return error






