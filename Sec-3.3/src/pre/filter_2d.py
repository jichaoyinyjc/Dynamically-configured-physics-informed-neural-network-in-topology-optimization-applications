import numpy as np
from scipy.sparse import coo_matrix


def get_ft(nex, ney, r_ft):
    n_ft = int(nex * ney * ((2 * (np.ceil(r_ft) - 1)) + 1) ** 2)
    iH = np.zeros(n_ft, dtype=int)
    jH = np.zeros(n_ft, dtype=int)
    sH = np.zeros(n_ft, dtype=float)
    cc = 0
    for i in range(nex):
        for j in range(ney):
            row = i * ney + j
            kk1 = int(np.maximum(i - (np.ceil(r_ft) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(r_ft), nex))
            ll1 = int(np.maximum(j - (np.ceil(r_ft) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(r_ft), ney))
            for s in range(kk1, kk2):
                for t in range(ll1, ll2):
                    col = s * ney + t
                    fac = r_ft - np.sqrt(((i - s) * (i - s) + (j - t) * (j - t)))
                    if fac > 0.:
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH[0:cc], (iH[0:cc], jH[0:cc])), shape=(nex * ney, nex * ney)).tocsc()
    Hs = H.sum(1)
    return H, Hs


def heaviside_ft(beta, rho_mid):
    # heaviside function, details refer to paper.  -- DOI: 10.1007/s00158-010-0602-y
    bound_low = 0.
    bound_upp = 1.
    eta = None
    rho_phys = None
    while bound_upp - bound_low > 1.0e-10:
        eta = 0.5 * (bound_low + bound_upp)
        rho_phys = (np.tanh(beta * eta) + np.tanh(beta * (rho_mid - eta))) / (
                np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
        if np.sum(rho_phys) >= np.sum(rho_mid):
            bound_low = eta
        else:
            bound_upp = eta
    return eta, rho_phys
