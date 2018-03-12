
cdef void make_n_basis_template(ns_type next_state,basis_type[:] ns_pars,npy_intp MAX,basis_type s,basis_type[:] basis):
    cdef npy_intp i
    for i in range(MAX):
        basis[i] = s
        s = next_state(s,ns_pars)


cdef npy_intp make_p_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX, basis_type s,
                                            int L,int pblock, N_type  * N, basis_type[:]  basis):
    cdef npy_intp Ns
    cdef int rp

    cdef int j


    Ns = 0

    while(MAX!=0):
        rp = CheckState_P_template(pblock,s,L,ns_pars)
        if rp > 0:
            basis[Ns] = s
            N[Ns] = rp
            Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_p_z_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                    npy_intp MAX,basis_type s,int L, int pblock, int zblock,
                                    N_type  * N, basis_type[:]  basis):
    cdef npy_intp Ns
    cdef int rpz


    Ns = 0
    while(MAX!=0):
        rpz = CheckState_P_Z_template(pblock,zblock,s,L,ns_pars)
        if rpz > 0:
            basis[Ns] = s
            N[Ns] = rpz
            Ns += 1
        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_pz_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                        npy_intp MAX,basis_type s,int L, int pzblock,
                                        N_type  * N, basis_type[:]  basis):
    cdef npy_intp Ns
    cdef int rpz


    Ns = 0
    while(MAX!=0):
        rpz = CheckState_PZ_template(pzblock,s,L,ns_pars)
        if rpz > 0:
            basis[Ns] = s
            N[Ns] = rpz
            Ns += 1
        
        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_t_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                        npy_intp MAX,basis_type s,int L,
                                        int kblock,int a,N_type  * N, basis_type[:]  basis):
    cdef npy_intp Ns
    cdef int r


    Ns = 0
    while(MAX!=0):
        r=CheckState_T_template(kblock,L,s,a,ns_pars)
        if r > 0:
            N[Ns] = r                
            basis[Ns] = s
            Ns += 1        

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_t_p_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int pblock,int kblock,int a, 
                                            N_type *N,M1_type *m,basis_type[:] basis):
    cdef npy_intp Ns
    cdef int r_temp,r,mp,sign
    cdef int sigma,sigma_i,sigma_f,v
    cdef int R[3]

    cdef double k = (2.0*_np.pi*kblock*a)/L

    if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
        sigma_i = 1
        sigma_f = 1
    else:
        sigma_i = -1
        sigma_f = 1

    Ns = 0

    R[0] = 0
    R[1] = 0
    R[2] = 0

    while(MAX!=0):
        CheckState_T_P_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mp = R[1]
        sign = R[2]
        if r > 0:
            if mp != -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if 1 + sign*sigma*pblock*cos(mp*k) == 0:
                        r_temp = -1
                    if (sigma == -1) and (1 - sign*sigma*pblock*cos(mp*k) != 0):
                        r_temp = -1
    
                    if r_temp > 0:
                        m[Ns] = mp + 1 + (L+1) * ((sign+1)//2)
                        N[Ns] = (sigma*r)                
                        basis[Ns] = s
                        Ns += 1
            else:
                for sigma in range(sigma_i,sigma_f+1,2):
                    m[Ns] = 0
                    N[Ns] = (sigma*r)                
                    basis[Ns] = s
                    Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns






cdef npy_intp make_t_p_z_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                                npy_intp MAX,basis_type s,
                                                int L, int pblock, int zblock, int kblock, int a,
                                                N_type *N, M2_type *m, basis_type[:] basis):
    cdef double k = 2.0*_np.pi*kblock*a/L
    cdef npy_intp Ns=0

    cdef int r,r_temp,mp,mz,mpz,sign1,sign2
    cdef int sigma,sigma_i,sigma_f
    cdef int R[7]

    if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
        sigma_i = 1
        sigma_f = 1
    else:
        sigma_i = -1
        sigma_f = 1

    cdef int j

    R[0] = 0
    R[1] = 0
    R[2] = 0
    R[3] = 0
    R[4] = 0
    R[5] = 0
    R[6] = 0

    while(MAX!=0):
        CheckState_T_P_Z_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mp = R[1]
        mz = R[2]
        mpz = R[3]
        signp = R[4]
        signz = R[5]
        signpz = R[6]
        signp1 = (R[4]+1)//2
        signz1 = (R[5]+1)//2
        signpz1 = (R[6]+1)//2
        if r>0:
            if mp == -1 and mz == -1 and mpz == -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    m[Ns] = 0 # c = 0
                    N[Ns] = (sigma*r)                
                    basis[Ns] = s
                    Ns += 1
            if mp != -1 and mz == -1 and mpz == -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if 1 + sigma*signp*pblock*cos(mp*k) == 0:
                        r_temp = -1
                    if (sigma == -1) and (1 - sigma*signp*pblock*cos(mp*k) != 0):
                        r_temp = -1
                    if r_temp > 0:

                        m[Ns] = mp + (1+signp1)*(L+1)**2 # c = 1 or 2 for + or - sign
                        N[Ns] = (sigma*r)                
                        basis[Ns] = s
                        Ns += 1
            if mp == -1 and mz != -1 and mpz == -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if 1 + zblock*signz*cos(k*mz) == 0:
                        r_temp = -1
                    if r_temp > 0:        
                        m[Ns] = mz*(L+1) + (3+signz1)*(L+1)**2 # c = 3 or 4 for + or - sign
                        N[Ns] = (sigma*r)                
                        basis[Ns] = s
                        Ns += 1
            if mp == -1 and mz == -1 and mpz != -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if 1 + sigma*signpz*pblock*zblock*cos(mpz*k) == 0:
                        r_temp = -1
                    if (sigma == -1) and (1 - sigma*signpz*pblock*zblock*cos(mpz*k) != 0):
                        r_temp = -1
                    if r_temp>0:
                        m[Ns] = mpz + (5+signpz1)*(L+1)**2 # c = 5 or 6 for + or - sign
                        N[Ns] = (sigma*r)                
                        basis[Ns] = s
                        Ns += 1
            if mp != -1 and mz != -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if (1 + sigma*signp*pblock*cos(mp*k) == 0) or (1 + zblock*signz*cos(mz*k) == 0):
                        r_temp = -1
                    if (sigma == -1) and ( (1 - sigma*signp*pblock*cos(mp*k) != 0) or (1 - zblock*signz*cos(mz*k) != 0) ):
                        r_temp = -1
                    if r_temp>0:
                        m[Ns] = mp + (L+1)*mz + (7+signp1+2*signz1)*(L+1)**2
                        N[Ns] = (sigma*r)                
                        basis[Ns] = s
                        Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_t_pz_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int pzblock,int kblock,int a,
                                            N_type *N,M1_type *m,basis_type[:] basis):
    cdef double k 
    
    cdef npy_intp Ns

    cdef int sigma,sigma_i,sigma_f
    cdef int r_temp,r,mpz,sign
    cdef int R[3]
    cdef int j
    
    k = 2.0*_np.pi*kblock*a/L

    if (2*kblock*a) % L == 0: #picks up k = 0, pi modes
        sigma_i = 1
        sigma_f = 1
    else:
        sigma_i = -1
        sigma_f = 1

    Ns = 0
    R[0] = 0
    R[1] = 0
    R[2] = 0
    while(MAX!=0):
        CheckState_T_PZ_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mpz = R[1]
        sign = R[2]
        if r > 0:
            if mpz != -1:
                for sigma in range(sigma_i,sigma_f+1,2):
                    r_temp = r
                    if 1 + sign*sigma*pzblock*cos(mpz*k) == 0:
                        r_temp = -1
                    if (sigma == -1) and (1 - sign*sigma*pzblock*cos(mpz*k) != 0):
                        r_temp = -1
    
                    if r_temp > 0:
                        m[Ns] = mpz + 1 + (L+1) * ((sign+1)//2)
                        N[Ns] = (sigma*r)
                        basis[Ns] = s
                        Ns += 1
            else:
                for sigma in range(sigma_i,sigma_f+1,2):
                    m[Ns] = 0
                    N[Ns] = (sigma*r)                
                    basis[Ns] = s
                    Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns






cdef npy_intp make_t_zA_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                                npy_intp MAX,basis_type s,
                                                int L,int zAblock,int kblock,int a,
                                                N_type *N, M1_type *m, basis_type[:] basis): 
    cdef double k 

    cdef npy_intp Ns
    cdef int mzA,r
    cdef int R[2]
    cdef int j    

    k = 2.0*_np.pi*kblock*a/L
    Ns = 0
    R[0] = 0
    R[0] = 0    

    while(MAX!=0):
        CheckState_T_ZA_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mzA = R[1]

        if r > 0:
            if mzA != -1:
                if 1 + zAblock*cos(k*mzA) == 0:
                    r = -1                

            if r > 0:
                m[Ns] = mzA + 1
                N[Ns] = r            
                basis[Ns] = s
                Ns += 1    

        s = next_state(s,ns_pars)
        MAX -= 1
       

    return Ns






cdef npy_intp make_t_zA_zB_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int zAblock,int zBblock,int kblock,int a,
                                            N_type *N,M2_type *m,basis_type[:] basis): 
    cdef double k 
    cdef npy_intp Ns
    cdef int mzA,mzB,mz,r

    cdef int R[4]
    cdef int j
    
    k = 2.0*_np.pi*kblock*a/L
    Ns = 0
    R[0] = 0
    R[1] = 0
    R[2] = 0
    R[3] = 0
    
    while(MAX!=0):
        CheckState_T_ZA_ZB_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mzA = R[1]
        mzB = R[2]
        mz = R[3]

        if r > 0:
            if mzA == -1 and mzB == -1 and mz == -1:
                m[Ns] = (L+1)
                N[Ns] = r            
                basis[Ns] = s
                Ns += 1    

            if mzA != -1 and mzB == -1 and mz == -1:
                if 1 + zAblock*cos(k*mzA) != 0:            
                    m[Ns] = mzA + 2*(L+1)
                    N[Ns] = r            
                    basis[Ns] = s
                    Ns += 1    

            if mzA == -1 and mzB != -1 and mz == -1:
                if 1 + zBblock*cos(k*mzB) != 0:        
                    m[Ns] = mzB + 3*(L+1)
                    N[Ns] = r            
                    basis[Ns] = s
                    Ns += 1

            if mzA == -1 and mzB == -1 and mz != -1:
                if 1 + zAblock*zBblock*cos(k*mz) != 0:                    
                    m[Ns] = mz + 4*(L+1)
                    N[Ns] = r            
                    basis[Ns] = s
                    Ns += 1
        
        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns





cdef npy_intp make_t_z_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int zblock,int kblock,int a,
                                            N_type *N,M1_type *m,basis_type[:] basis): 
    cdef double k 
    cdef npy_intp Ns

    cdef int mz,r,sign
    cdef int R[3]
    cdef int j
    

    k = 2.0*_np.pi*kblock*a/L


    Ns = 0
    R[0] = 0
    R[1] = 0
    R[2] = 0

    while(MAX!=0):
        CheckState_T_Z_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mz = R[1]
        sign = R[2]

        if mz != -1:
            if 1 + sign*zblock*cos(k*mz) == 0:
                r = -1
        else:
            sign = -1          

        if r > 0:
            m[Ns] = mz + 1 + ((sign+1)//2) * (L+1)
            N[Ns] = r
            basis[Ns] = s
            Ns += 1    
        
        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns




    

cdef npy_intp make_t_zB_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int zBblock,int kblock,int a,
                                            N_type *N, M1_type *m,basis_type[:] basis): 
    cdef double k 
    cdef npy_intp Ns

    cdef int mzB,r
    cdef int R[2]
    cdef int j
    

    k = 2.0*_np.pi*kblock*a/L
    Ns = 0
    R[0] = 0
    R[1] = 0
    
    while(MAX!=0):
        CheckState_T_ZB_template(kblock,L,s,a,R,ns_pars)
        r = R[0]
        mzB = R[1]

        if r > 0:
            if mzB != -1:
                if 1 + zBblock*cos(k*mzB) == 0:
                    r = -1                

            if r > 0:
                m[Ns] = mzB + 1
                N[Ns] = r            
                basis[Ns] = s
                Ns += 1    
        
        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns


cdef npy_intp make_z_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                        npy_intp MAX,basis_type s,
                                        int L,int zblock,N_type *N,basis_type[:] basis):
    cdef npy_intp Ns
    cdef int rz
    cdef int j


    Ns = 0
    while(MAX!=0):
        rz = CheckState_Z_template(zblock,s,L,ns_pars)
        if rz > 0:
            basis[Ns] = s
            N[Ns] = rz
            Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns



cdef npy_intp make_zA_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int zAblock, N_type *N, basis_type[:] basis):
    cdef npy_intp Ns
    cdef int rzA
    cdef int j


    Ns = 0
    while(MAX!=0):
        rzA = CheckState_ZA_template(zAblock,s,L,ns_pars)
        if rzA > 0:
            basis[Ns] = s
            N[Ns] = rzA
            Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns


cdef npy_intp make_zB_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                            npy_intp MAX,basis_type s,
                                            int L,int zBblock,N_type *N, basis_type[:] basis):
    cdef npy_intp Ns
    cdef int rzB
    cdef int j


    Ns = 0
    while(MAX!=0):
        rzB = CheckState_ZB_template(zBblock,s,L,ns_pars)
        if rzB > 0:
            basis[Ns] = s
            N[Ns] = rzB
            Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns




cdef npy_intp make_zA_zB_basis_template(ns_type next_state, basis_type[:] ns_pars,
                                        npy_intp MAX,basis_type s,
                                        int L,int zAblock,int zBblock,N_type *N,basis_type[:] basis):
    cdef npy_intp Ns

    cdef int r
    cdef int j

    Ns = 0
    while(MAX!=0):
        r = CheckState_ZA_ZB_template(zAblock,zBblock,s,L,ns_pars)
        if r > 0:
            basis[Ns] = s
            N[Ns] = r
            Ns += 1

        s = next_state(s,ns_pars)
        MAX -= 1


    return Ns
