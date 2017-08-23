

cdef int op_template(op_type op_func, basis_type[:] op_pars, npy_intp Ns, basis_type[:] basis,
                     str opstr, NP_INT32_t *indx, scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef npy_intp i
    cdef int error

    for i in range(Ns):
        col[i] = i

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    for i in range(Ns):
        row[i] = Ns - row[i] - 1

    return error




cdef int n_op_template(op_type op_func,basis_type[:] op_pars, npy_intp Ns, basis_type[:] basis,
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
        row[i] = findzstate(basis,Ns,row[i],&found)
        if not found:
            ME[i] = _np.nan


    return error








cdef int p_op_template(op_type op_func,basis_type[:] op_pars,bitop fliplr,basis_type[:] ref_pars,int L,int pblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):

    cdef basis_type s
    cdef npy_intp i
    cdef int error = 0
    cdef int q
    cdef double n
    cdef bool found

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_P_template(fliplr,row[i],L,&q,ref_pars)
        s = findzstate(basis,Ns,s,&found)
        
        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= (pblock**q)*sqrt(n)

    return error




cdef int pz_op_template(op_type op_func,basis_type[:] op_pars,bitop fliplr,bitop flip_all,basis_type[:] ref_pars,int L,int pzblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,qg
    cdef double n
    cdef bool found

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_PZ_template(fliplr,flip_all,row[i],L,&qg,ref_pars)
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]

        ME[i] *= (pzblock**qg)*sqrt(n)

    return error











cdef int p_z_op_template(op_type op_func,basis_type[:] op_pars,bitop fliplr,bitop flip_all,basis_type[:] ref_pars,int L,int pblock,int zblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,q,g
    cdef int R[2]
    cdef double n
    cdef bool found

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_P_Z_template(fliplr,flip_all,row[i],L,R,ref_pars)

        q = R[0]
        g = R[1]
        
        s = findzstate(basis,Ns,s,&found)
        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]

        ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

    return error






















cdef int t_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,basis_type[:] ref_pars,int L,int kblock,int a, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,l
    cdef double n,k
    cdef bool found

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
        s = RefState_T_template(shift,row[i],L,a,&l,ref_pars)
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[i]
        n /= N[s]
        n = sqrt(n)
        ME[i] *= n

        if (matrix_type is float or matrix_type is double):
            ME[i] *= (-1.0)**(l*2*a*kblock/L)
        else:
            ME[i] *= (cos(k*l) - 1.0j * sin(k*l))

            

    return error






cdef double MatrixElement_T_P(int L,int pblock, int kblock, int a, int l, double k, int q,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double ME
    cdef int sr,sc

    if Nr > 0:
        sr = 1
    else:
        sr = -1

    if Nc > 0:
        sc = 1
    else:
        sc = -1


    if mr >= 0:
        nr = (1 + sr*pblock*cos(k*mr))/Nr
    else:
        nr = 1.0/Nr
    nr *= sr

    if mc >= 0:
        nc = (1 + sc*pblock*cos(k*mc))/Nc
    else:
        nc = 1.0/Nc
    nc *= sc


    ME=sqrt(nc/nr)*(sr*pblock)**q

    if sr == sc :
        if mc < 0:
            ME *= cos(k*l)
        else:
            ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
    else:
        if mc < 0:
            ME *= -sr*sin(k*l)
        else:
            ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))        


    return ME






cdef int t_p_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop fliplr,basis_type[:] ref_pars,int L,int kblock,int pblock,int a, npy_intp Ns,
                        N_type *N, N_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,q
    cdef int R[2]
    cdef bool found

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
            s = RefState_T_P_template(shift,fliplr,row[i],L,a,R,ref_pars)

            l = R[0]
            q = R[1]

            ss = findzstate(basis,Ns,s,&found)

            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


    else:
        for i in range(Ns):
            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            s = RefState_T_P_template(shift,fliplr,row[i],L,a,R,ref_pars)

            l = R[0]
            q = R[1]

            ss = findzstate(basis,Ns,s,&found)

            if not found:
                for j in range(i,i+o,1):
                    ME[j] = _np.nan
                continue


            if (ss == i) and (q == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
                        ME[j] = me[c]*MatrixElement_T_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error




cdef double MatrixElement_T_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, double k, int q, int g,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double ME
    cdef int sr,sc
    cdef int nnr,mmr,cr,nnc,mmc,cc

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
    nnc = (mc/(L+1)) % (L+1)
    cc = mc/((L+1)*(L+1))

    mmr = mr % (L+1)
    nnr = (mr/(L+1)) % (L+1)
    cr = mr/((L+1)*(L+1))

    nr = 1.0
    nc = 1.0

    if cr == 1:
        nr = 1.0/Nr
    elif cr == 2:
        nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
    elif cr == 3:
        nr = (1.0 + zblock*cos(k*nnr))/Nr
    elif cr == 4:
        nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr    
    elif cr == 5:
        nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

    nr *= sr

    if cc == 1:
        nc = 1.0/Nc
    elif cc == 2:
        nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
    elif cc == 3:
        nc = (1.0 + zblock*cos(k*nnc))/Nc
    elif cc == 4:
        nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc    
    elif cc == 5:
        nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

    nc *= sc



    ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

    if sr == sc :
        if (cc == 1) or (cc == 3):
            ME *= cos(k*l)
        elif (cc == 2) or (cc == 5):
            ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
        elif (cc == 4):
            ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
    else:
        if (cc == 1) or (cc == 3):
            ME *= -sr*sin(k*l)
        elif (cc == 2) or (cc == 5):
            ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
        elif (cc == 4):
            ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

    return ME






cdef int t_p_z_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop fliplr,bitop flip_all,basis_type[:] ref_pars,int L,int kblock,int pblock,int zblock,int a,
                        npy_intp Ns, N_type *N, M_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx,
                        scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,q,g
    cdef int R[3]
    cdef bool found

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
            s = RefState_T_P_Z_template(shift,fliplr,flip_all,row[i],L,a,R,ref_pars)

            l = R[0]
            q = R[1]
            g = R[2]

            ss = findzstate(basis,Ns,s,&found)


            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])

    else:
        for i in range(Ns):
            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            s = RefState_T_P_Z_template(shift,fliplr,flip_all,row[i],L,a,R,ref_pars)

            l = R[0]
            q = R[1]
            g = R[2]

            ss = findzstate(basis,Ns,s,&found)

            if not found:
                for j in range(i,i+o,1):
                    ME[j]  = _np.nan
                continue


            if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
                        ME[j] = me[c]*MatrixElement_T_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error







cdef double MatrixElement_T_PZ(int L,int pzblock, int kblock, int a, int l, double k, int qg,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double ME
    cdef int sr,sc



    if Nr > 0:
        sr = 1
    else:
        sr = -1

    if Nc > 0:
        sc = 1
    else:
        sc = -1




    if mr >= 0:
        nr = (1 + sr*pzblock*cos(k*mr))/Nr
    else:
        nr = 1.0/Nr
    nr *= sr

    if mc >= 0:
        nc = (1 + sc*pzblock*cos(k*mc))/Nc
    else:
        nc = 1.0/Nc
    nc *= sc

    ME=sqrt(nc/nr)*(sr*pzblock)**qg

    if sr == sc :
        if mc < 0:
            ME *= cos(k*l)
        else:
            ME *= (cos(k*l)+sr*pzblock*cos((l-mc)*k))/(1+sr*pzblock*cos(k*mc))
    else:
        if mc < 0:
            ME *= -sr*sin(k*l)
        else:
            ME *= (-sr*sin(k*l)+pzblock*sin((l-mc)*k))/(1-sr*pzblock*cos(k*mc))        


    return ME






cdef int t_pz_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop fliplr,bitop flip_all,basis_type[:] ref_pars,int L,int kblock,int pzblock,int a,
                        npy_intp Ns, N_type *N, N_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx,
                        scalar_type J, basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i,j,o,p,c,b
    cdef matrix_type me[2]
    cdef int error,l,qg
    cdef int R[2]
    cdef bool found

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
            s = RefState_T_PZ_template(shift,fliplr,flip_all,row[i],L,a,R,ref_pars)

            l = R[0]
            qg = R[1]

            ss = findzstate(basis,Ns,s,&found)


            if not found:
                ME[i] = _np.nan
                continue

            row[i] = ss
            ME[i] *= MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


    else:
        for i in range(Ns):
            if (i > 0) and (basis[i] == basis[i-1]): continue
            if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
                o = 2
            else:
                o = 1

            s = RefState_T_PZ_template(shift,fliplr,flip_all,row[i],L,a,R,ref_pars)

            l = R[0]
            qg = R[1]

            ss = findzstate(basis,Ns,s,&found)

            if not found:
                for j in range(i,i+o,1):
                    ME[j] = _np.nan
                continue


            if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

                for j in range(i,i+o,1):
                    row[j] = j
                    ME[j] *= MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

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
                        ME[j] = me[c]*MatrixElement_T_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
        
        
    return error








cdef double complex MatrixElement_T_ZA(int L,int zAblock, int kblock, int a, int l, double k, int gA,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    

    if mr >=0:
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



cdef int t_zA_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop flip_sublat_A,basis_type[:] ref_pars,int L,int kblock,int zAblock,int a, npy_intp Ns,
                        N_type *N, N_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,gA
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found

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
        s = RefState_T_ZA_template(shift,flip_sublat_A,row[i],L,a,R,ref_pars)

        l = R[0]
        gA = R[1]

        ss = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue


        row[i] = ss


        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZA(L,zAblock,kblock,a,l,k,gA,N[i],N[ss],m[i],m[ss])
            ME[i] *= me.real
        else:
            ME[i] *= MatrixElement_T_ZA(L,zAblock,kblock,a,l,k,gA,N[i],N[ss],m[i],m[ss])

        
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



cdef int t_zA_zB_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop flip_sublat_A,bitop flip_sublat_B,bitop flip_all,basis_type[:] ref_pars,
                        int L,int kblock,int zAblock,int zBblock,int a, npy_intp Ns, N_type *N, M_type *m,
                        basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,gA,gB
    cdef int R[3]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found

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
        s = RefState_T_ZA_ZB_template(shift,flip_sublat_A,flip_sublat_B,flip_all,row[i],L,a,R,ref_pars)

        l = R[0]
        gA = R[1]
        gB = R[2]

        ss = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
            ME[i] *= me.real
        else:
            ME[i] *= MatrixElement_T_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])


    return error









cdef double complex MatrixElement_T_ZB(int L,int zBblock, int kblock, int a, int l, double k, int gB,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    

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



cdef int t_zB_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop flip_sublat_B,basis_type[:] ref_pars,int L,int kblock,int zBblock,int a, npy_intp Ns,
                        N_type *N, N_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i 
    cdef int error,l,gB
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found

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
        s = RefState_T_ZB_template(shift,flip_sublat_B,row[i],L,a,R,ref_pars)

        l = R[0]
        gB = R[1]

        ss = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])
            ME[i] *= me.real
        else:
            ME[i] *= MatrixElement_T_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])


    return error





cdef double complex MatrixElement_T_Z(int L,int zblock, int kblock, int a, int l, double k, int g,int Nr,int Nc,int mr,int mc):
    cdef double nr,nc
    cdef double complex ME
    

    if mr >=0:
        nr = (1 + zblock*cos(k*mr))/Nr
    else:
        nr = 1.0/Nr

    if mc >= 0:
        nc = (1 + zblock*cos(k*mc))/Nc
    else:
        nc = 1.0/Nc


    ME=sqrt(nc/nr)*(zblock**g)

    if ((2*a*kblock) % L) == 0:
        ME *= (-1)**(2*l*a*kblock/L)
    else:
        ME *= (cos(k*l) - 1.0j * sin(k*l))

    return ME



cdef int t_z_op_template(op_type op_func,basis_type[:] op_pars,shifter shift,bitop flip_all,basis_type[:] ref_pars,int L,int kblock,int zblock,int a, npy_intp Ns,
                        N_type *N, N_type *m, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,l,g
    cdef int R[2]
    cdef double k = (2.0*_np.pi*kblock*a)/L
    cdef double complex me
    cdef bool found

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
        s = RefState_T_Z_template(shift,flip_all,row[i],L,a,R,ref_pars)

        l = R[0]
        g = R[1]

        ss = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = ss

        if (matrix_type is float or matrix_type is double):
            me = MatrixElement_T_Z(L,zblock,kblock,a,l,k,g,N[i],N[ss],m[i],m[ss])
            ME[i] *= me.real
        else:
            ME[i] *= MatrixElement_T_Z(L,zblock,kblock,a,l,k,g,N[i],N[ss],m[i],m[ss])
            

    return error








cdef int zA_op_template(op_type op_func, basis_type[:] op_pars,bitop flip_sublat_A,basis_type[:] ref_pars,int L,int zAblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,gA
    cdef bool found
    cdef double n

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_ZA_template(flip_sublat_A,row[i],L,&gA,ref_pars)
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= (zAblock)**gA*sqrt(n)


    return error







cdef int zA_zB_op_template(op_type op_func,basis_type[:] op_pars,bitop flip_sublat_A,bitop flip_sublat_B,bitop flip_all,basis_type[:] ref_pars,
                        int L,int zAblock,int zBblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,gA,gB
    cdef int R[2]
    cdef bool found
    cdef double n

    R[0] = 0
    R[1] = 0

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_ZA_ZB_template(flip_sublat_A,flip_sublat_B,flip_all,row[i],L,R,ref_pars)

        gA = R[0]
        gB = R[1]
        
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= (zAblock**gA)*(zBblock**gB)*sqrt(n)

    return error



cdef int zB_op_template(op_type op_func,basis_type[:] op_pars,bitop flip_sublat_B,basis_type[:] ref_pars,int L,int zBblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp ss,i
    cdef int error,gB
    cdef bool found
    cdef double n

    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_ZB_template(flip_sublat_B,row[i],L,&gB,ref_pars)
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= (zBblock)**gB*sqrt(n)


    return error



cdef int z_op_template(op_type op_func,basis_type[:] op_pars,bitop flip_all,basis_type[:] ref_pars,int L,int zblock, npy_intp Ns,
                        N_type *N, basis_type[:] basis, str opstr, NP_INT32_t *indx, scalar_type J,
                        basis_type[:] row, basis_type[:] col, matrix_type *ME):
    cdef basis_type s
    cdef npy_intp i
    cdef int error,g
    cdef bool found
    cdef double n


    error = op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)

    if error != 0:
        return error

    for i in range(Ns):
        col[i] = i

    for i in range(Ns):
        s = RefState_Z_template(flip_all,row[i],L,&g,ref_pars)
        s = findzstate(basis,Ns,s,&found)

        if not found:
            ME[i] = _np.nan
            continue

        row[i] = s
        n =  N[s]
        n /= N[i]
        ME[i] *= (zblock)**g*sqrt(n)


    return error






