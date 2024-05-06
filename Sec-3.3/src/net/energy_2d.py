import torch
import numpy as np
import random

random.seed(10000)
np.random.seed(10000)
torch.manual_seed(10000)
torch.cuda.manual_seed(10000)

# pre-process for calculating internal energy
def pre_internal(ele_geo):
    ex = ele_geo[0]
    ey = ele_geo[1]
    #
    gp_values = 1. / np.sqrt(3.)
    gp1_x, gp1_y = (-gp_values, -gp_values)
    gp2_x, gp2_y = (-gp_values, gp_values)
    gp3_x, gp3_y = (gp_values, gp_values)
    gp4_x, gp4_y = (gp_values, -gp_values)
    ''' 
    the partial derivative of shape function with respect to s (horizontal coordinate),
    [dNi_ds|gp1, dNi_ds|gp2, dNi_ds|gp3, dNi_ds|gp4] 
    '''
    dN1_ds = 0.25 * torch.tensor([[-(1 - gp1_y), -(1 - gp2_y), -(1 - gp3_y), -(1 - gp4_y)]])
    dN2_ds = 0.25 * torch.tensor([[-(1 + gp1_y), -(1 + gp2_y), -(1 + gp3_y), -(1 + gp4_y)]])
    dN3_ds = 0.25 * torch.tensor([[(1 + gp1_y), (1 + gp2_y), (1 + gp3_y), (1 + gp4_y)]])
    dN4_ds = 0.25 * torch.tensor([[(1 - gp1_y), (1 - gp2_y), (1 - gp3_y), (1 - gp4_y)]])
    '''
    the partial derivative of shape function with respect to t (vertical coordinate)
    [dNi_dt|gp1, dNi_dt|gp2, dNi_dt|gp3, dNi_dt|gp4]
    '''
    dN1_dt = 0.25 * torch.tensor([[-(1 - gp1_x), -(1 - gp2_x), -(1 - gp3_x), -(1 - gp4_x)]])
    dN2_dt = 0.25 * torch.tensor([[(1 - gp1_x), (1 - gp2_x), (1 - gp3_x), (1 - gp4_x)]])
    dN3_dt = 0.25 * torch.tensor([[(1 + gp1_x), (1 + gp2_x), (1 + gp3_x), (1 + gp4_x)]])
    dN4_dt = 0.25 * torch.tensor([[-(1 + gp1_x), -(1 + gp2_x), -(1 + gp3_x), -(1 + gp4_x)]])
    '''
    nodes_x = torch.tensor([[-ex/2.], [-ex/2.], [+ex/2.], [+ex/2.]])
    nodes_y = torch.tensor([[-ey/2.], [+ey/2.], [+ey/2.], [-ey/2.]])
    J1 = torch.tensor([[dN1_ds @ nodes_x, dN1_ds @ nodes_y], [dN1_dt @ nodes_x, dN1_dt @ nodes_y]])
    J2 = torch.tensor([[dN2_ds @ nodes_x, dN2_ds @ nodes_y], [dN2_dt @ nodes_x, dN2_dt @ nodes_y]])
    J3 = torch.tensor([[dN3_ds @ nodes_x, dN3_ds @ nodes_y], [dN3_dt @ nodes_x, dN3_dt @ nodes_y]])
    J4 = torch.tensor([[dN4_ds @ nodes_x, dN4_ds @ nodes_y], [dN4_dt @ nodes_x, dN4_dt @ nodes_y]])
    jacobi_mat = J1 = J2 = J3 = J4, and jacobi_mat has no rounding error
    '''
    jacobi = torch.tensor([[ex / 2., 0.], [0., ey / 2.]]).double()
    jacobi_inv = torch.linalg.inv(jacobi)

    ''' strain at gauss point   dNi_dxy = [dNi_dx, [dNi_dy].T '''
    dN1_dxy = jacobi_inv @ torch.tensor([[dN1_ds[0, 0], dN1_ds[0, 1], dN1_ds[0, 2], dN1_ds[0, 3]],
                                         [dN1_dt[0, 0], dN1_dt[0, 1], dN1_dt[0, 2], dN1_dt[0, 3]]])
    dN2_dxy = jacobi_inv @ torch.tensor([[dN2_ds[0, 0], dN2_ds[0, 1], dN2_ds[0, 2], dN2_ds[0, 3]],
                                         [dN2_dt[0, 0], dN2_dt[0, 1], dN2_dt[0, 2], dN2_dt[0, 3]]])
    dN3_dxy = jacobi_inv @ torch.tensor([[dN3_ds[0, 0], dN3_ds[0, 1], dN3_ds[0, 2], dN3_ds[0, 3]],
                                         [dN3_dt[0, 0], dN3_dt[0, 1], dN3_dt[0, 2], dN3_dt[0, 3]]])
    dN4_dxy = jacobi_inv @ torch.tensor([[dN4_ds[0, 0], dN4_ds[0, 1], dN4_ds[0, 2], dN4_ds[0, 3]],
                                         [dN4_dt[0, 0], dN4_dt[0, 1], dN4_dt[0, 2], dN4_dt[0, 3]]])
    return dN1_dxy, dN2_dxy, dN3_dxy, dN4_dxy


# get internal energy
def get_internal(u, ele_geo, density, em, nu, simp_p, flag, id_node, dN1_dxy, dN2_dxy, dN3_dxy, dN4_dxy):
    u_xN1, u_yN1 = u[id_node[:, 0], 0], u[id_node[:, 0], 1]
    u_xN2, u_yN2 = u[id_node[:, 1], 0], u[id_node[:, 1], 1]
    u_xN3, u_yN3 = u[id_node[:, 2], 0], u[id_node[:, 2], 1]
    u_xN4, u_yN4 = u[id_node[:, 3], 0], u[id_node[:, 3], 1]
    ex = ele_geo[0]
    ey = ele_geo[1]
    jacobi = torch.tensor([[ex / 2., 0.], [0., ey / 2.]]).double()
    det_jacobi = torch.linalg.det(jacobi)
    # strain and stress at gauss point 1
    e_xx_gp1 = dN1_dxy[0, 0] * u_xN1 + dN2_dxy[0, 0] * u_xN2 + dN3_dxy[0, 0] * u_xN3 + dN4_dxy[0, 0] * u_xN4
    e_yy_gp1 = dN1_dxy[1, 0] * u_yN1 + dN2_dxy[1, 0] * u_yN2 + dN3_dxy[1, 0] * u_yN3 + dN4_dxy[1, 0] * u_yN4
    e_xy_gp1 = 0.5 * (dN1_dxy[0, 0] * u_yN1 + dN2_dxy[0, 0] * u_yN2 +
                      dN3_dxy[0, 0] * u_yN3 + dN4_dxy[0, 0] * u_yN4 +
                      dN1_dxy[1, 0] * u_xN1 + dN2_dxy[1, 0] * u_xN2 +
                      dN3_dxy[1, 0] * u_xN3 + dN4_dxy[1, 0] * u_xN4)
    s_xx_gp1 = (em * (e_xx_gp1 + nu * e_yy_gp1) / (1 - nu * nu))
    s_yy_gp1 = (em * (e_yy_gp1 + nu * e_xx_gp1) / (1 - nu * nu))
    s_xy_gp1 = (em * e_xy_gp1 / (1 + nu))
    # strain and stress at gauss point 2
    e_xx_gp2 = dN1_dxy[0, 1] * u_xN1 + dN2_dxy[0, 1] * u_xN2 + dN3_dxy[0, 1] * u_xN3 + dN4_dxy[0, 1] * u_xN4
    e_yy_gp2 = dN1_dxy[1, 1] * u_yN1 + dN2_dxy[1, 1] * u_yN2 + dN3_dxy[1, 1] * u_yN3 + dN4_dxy[1, 1] * u_yN4
    e_xy_gp2 = 0.5 * (dN1_dxy[0, 1] * u_yN1 + dN2_dxy[0, 1] * u_yN2 +
                      dN3_dxy[0, 1] * u_yN3 + dN4_dxy[0, 1] * u_yN4 +
                      dN1_dxy[1, 1] * u_xN1 + dN2_dxy[1, 1] * u_xN2 +
                      dN3_dxy[1, 1] * u_xN3 + dN4_dxy[1, 1] * u_xN4)
    s_xx_gp2 = (em * (e_xx_gp2 + nu * e_yy_gp2) / (1 - nu * nu))
    s_yy_gp2 = (em * (e_yy_gp2 + nu * e_xx_gp2) / (1 - nu * nu))
    s_xy_gp2 = (em * e_xy_gp2 / (1 + nu))
    # strain and stress at gauss point 3
    e_xx_gp3 = dN1_dxy[0, 2] * u_xN1 + dN2_dxy[0, 2] * u_xN2 + dN3_dxy[0, 2] * u_xN3 + dN4_dxy[0, 2] * u_xN4
    e_yy_gp3 = dN1_dxy[1, 2] * u_yN1 + dN2_dxy[1, 2] * u_yN2 + dN3_dxy[1, 2] * u_yN3 + dN4_dxy[1, 2] * u_yN4
    e_xy_gp3 = 0.5 * (dN1_dxy[0, 2] * u_yN1 + dN2_dxy[0, 2] * u_yN2 +
                      dN3_dxy[0, 2] * u_yN3 + dN4_dxy[0, 2] * u_yN4 +
                      dN1_dxy[1, 2] * u_xN1 + dN2_dxy[1, 2] * u_xN2 +
                      dN3_dxy[1, 2] * u_xN3 + dN4_dxy[1, 2] * u_xN4)
    s_xx_gp3 = (em * (e_xx_gp3 + nu * e_yy_gp3) / (1 - nu * nu))
    s_yy_gp3 = (em * (e_yy_gp3 + nu * e_xx_gp3) / (1 - nu * nu))
    s_xy_gp3 = (em * e_xy_gp3 / (1 + nu))
    # strain and stress at gauss point 4
    e_xx_gp4 = dN1_dxy[0, 3] * u_xN1 + dN2_dxy[0, 3] * u_xN2 + dN3_dxy[0, 3] * u_xN3 + dN4_dxy[0, 3] * u_xN4
    e_yy_gp4 = dN1_dxy[1, 3] * u_yN1 + dN2_dxy[1, 3] * u_yN2 + dN3_dxy[1, 3] * u_yN3 + dN4_dxy[1, 3] * u_yN4
    e_xy_gp4 = 0.5 * (dN1_dxy[0, 3] * u_yN1 + dN2_dxy[0, 3] * u_yN2 +
                      dN3_dxy[0, 3] * u_yN3 + dN4_dxy[0, 3] * u_yN4 +
                      dN1_dxy[1, 3] * u_xN1 + dN2_dxy[1, 3] * u_xN2 +
                      dN3_dxy[1, 3] * u_xN3 + dN4_dxy[1, 3] * u_xN4)
    s_xx_gp4 = (em * (e_xx_gp4 + nu * e_yy_gp4) / (1 - nu * nu))
    s_yy_gp4 = (em * (e_yy_gp4 + nu * e_xx_gp4) / (1 - nu * nu))
    s_xy_gp4 = (em * e_xy_gp4 / (1 + nu))
    # strain energy
    se_gp1 = (e_xx_gp1 * s_xx_gp1 + e_yy_gp1 * s_yy_gp1 + 2 * e_xy_gp1 * s_xy_gp1)
    se_gp2 = (e_xx_gp2 * s_xx_gp2 + e_yy_gp2 * s_yy_gp2 + 2 * e_xy_gp2 * s_xy_gp2)
    se_gp3 = (e_xx_gp3 * s_xx_gp3 + e_yy_gp3 * s_yy_gp3 + 2 * e_xy_gp3 * s_xy_gp3)
    se_gp4 = (e_xx_gp4 * s_xx_gp4 + e_yy_gp4 * s_yy_gp4 + 2 * e_xy_gp4 * s_xy_gp4)
    strain_energy = 0.5 * det_jacobi * (se_gp1 + se_gp2 + se_gp3 + se_gp4)
    # Strain energy at element
    E_min_by_E_0 = 1.0e-6
    if flag == 'grad':
        # grad
        density_grad = (simp_p - simp_p * E_min_by_E_0) * torch.pow(density, simp_p - 1.)
        strain_energy_grad = strain_energy * density_grad
        # internal energy
        density_simp = E_min_by_E_0 + (1 - E_min_by_E_0) * torch.pow(density, simp_p)
        strain_energy_simp = strain_energy * density_simp
        return -strain_energy_grad, torch.sum(strain_energy_simp)
    elif flag == 'train':
        # internal energy
        density_simp = E_min_by_E_0 + (1 - E_min_by_E_0) * torch.pow(density, simp_p)
        strain_energy_simp = strain_energy * density_simp
        return torch.sum(strain_energy_simp)
    else:
        print("Error in training flag !!!")
        return 0


# get external energy
def get_external(u, load_value, id_load):
    W_ext = u[id_load, :] * load_value
    ext_energy = W_ext.sum()
    return ext_energy
