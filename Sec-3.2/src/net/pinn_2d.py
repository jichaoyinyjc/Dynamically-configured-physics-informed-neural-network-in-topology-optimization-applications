import torch
import numpy as np
from src.net.energy_2d import pre_internal, get_internal, get_external
from src.net.mlp_lib_2d import NN1, NN2
import random

random.seed(10000)
np.random.seed(10000)
torch.manual_seed(10000)
torch.cuda.manual_seed(10000)

if torch.cuda.is_available():
    device_string = 'cuda'
    dev = torch.device('cuda')
    print("\nCUDA is available, running on GPU")
else:
    device_string = 'cpu'
    dev = torch.device('cpu')
    print("\nCUDA not available, running on CPU")


class Pinn:
    def __init__(self, net_pars, net_2_pars, train_pars):
        super(Pinn, self).__init__()
        self.lr = train_pars["lr"]
        self.max_epoch = train_pars["max_epoch"]
        self.net_1 = NN1(
            net_pars["neurons"],
            net_pars["n_layers"],
            net_pars["rff_std"],
            net_pars["act_fn"],
        )
        self.net_1 = self.net_1.to(dev)
        self.net_2 = NN2(
            net_2_pars["neurons"],
            net_2_pars["n_layers"],
            net_2_pars["act_fn"])
        self.net_2 = self.net_2.to(dev)
        print(self.net_1)
        print(self.net_2)
        self.single_nn1 = False
        self.single_nn2 = False

    ''' set the bool variable: self.single_nn2_flag '''

    def set_flag_nn1(self, flag):
        self.single_nn1 = flag

    def set_flag_nn2(self, flag):
        self.single_nn2 = flag

    ''' training model '''

    def training(self, req_dom, id_node, req_rho, em, nu, simp_p, ele_geo, load, id_load, func_x, func_y):
        # collection points
        points = torch.from_numpy(req_dom)
        del req_dom
        points = points.to(dev)
        points.requires_grad_(True)

        # density
        density = torch.from_numpy(req_rho).to(dev)
        del req_rho

        # neumann boundary conditions
        load_value = torch.from_numpy(load['value']).to(dev)

        # minimizing loss function
        solver, max_epoch = None, None
        if self.single_nn1:
            solver = torch.optim.Adam(self.net_1.parameters(), lr=self.lr)
            max_epoch = self.max_epoch[0]
            print("Use solver net_1 and the max epoch is ", max_epoch)
        if self.single_nn2:
            solver = torch.optim.Adam(self.net_2.parameters(), lr=self.lr)
            max_epoch = self.max_epoch[1]
            print("Use solver net_2, and the max epoch is ", max_epoch)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(solver, gamma=0.999)
        loss_train = []
        # pre-process for calculating internal energy
        dN1_dxy, dN2_dxy, dN3_dxy, dN4_dxy = pre_internal(ele_geo)
        dN1_dxy = dN1_dxy.to(dev)
        dN2_dxy = dN2_dxy.to(dev)
        dN3_dxy = dN3_dxy.to(dev)
        dN4_dxy = dN4_dxy.to(dev)
        for epoch in range(max_epoch):
            u_pre = self.get_u(points, func_x, func_y)
            inE = get_internal(u_pre, ele_geo, density, em, nu, simp_p, 'train',
                               id_node, dN1_dxy, dN2_dxy, dN3_dxy, dN4_dxy)
            exE = get_external(u_pre, load_value, id_load)
            loss = (inE - exE)
            solver.zero_grad()
            loss.backward()
            solver.step()
            scheduler.step()
            loss_train.append(loss.item())
            # print('Iter: %d Loss: %.6e InE: %.4e ExE: %.4e' % (epoch + 1, loss.item(), inE.item(), exE.item()))
            # if epoch > 10:
            #     loss_std = np.std(loss_train[epoch-9:epoch+1])
            #     if epoch > 800 and loss_std.item() < 1e-4:
            #         print('Epoch: %d  loss_std: %.4e ' % (epoch + 1, loss_std.item()))
            #         break
        # stop training, output predict displacement of net
        u_pre_ = self.get_u(points, func_x, func_y)
        df_drp, comp = get_internal(u_pre_, ele_geo, density, em, nu, simp_p, 'grad',
                                    id_node, dN1_dxy, dN2_dxy, dN3_dxy, dN4_dxy)
        return df_drp, comp, u_pre_, loss_train

    ''' get displacement'''

    def get_u(self, points, func_x, func_y):
        points = points.float()  # input of rff.layers.GaussianEncoding must be float32
        u = self.net_1(points)
        u_x = (func_x(points) * u[:, 0])[np.newaxis].T
        u_y = (func_y(points) * u[:, 1])[np.newaxis].T
        if self.single_nn1:
            u_predict = torch.cat((u_x, u_y), dim=1).double()
            return u_predict
        if self.single_nn2:
            # u_for_fac = torch.cat((u_x, u_y), dim=1)
            # fac = self.net_2(u_for_fac)
            u_for_fac = torch.cat((u_x, u_y, points), dim=1)
            fac = self.net_2(u_for_fac)
            u_x = u_x * fac[:, 0][np.newaxis].T
            u_y = u_y * fac[:, 1][np.newaxis].T
            u_predict = torch.cat((u_x, u_y), dim=1).double()
            return u_predict
