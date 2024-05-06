import cv2
import time
import numpy as np
import pandas as pd
from src.pre.filter_2d import heaviside_ft
from src.pre.bcs_2d import SetBCs
from src.assist.plot_lib_2d import plot_dis, plot_sen_den, plot_time_obj
from src.opt.opt_solver_2d import topo_oc, topo_mma, set_mma


class Topo:
    def __init__(self, length, height, nex, ney, load_total, load_size):
        # geometry
        self.length = length
        self.height = height
        self.nex = nex
        self.ney = ney
        # boundary
        self.load_total = load_total
        self.load_size = load_size
        # geometry
        self.ex = length[0] / nex
        self.ey = length[0] / nex  # ex = ey
        self.ndx = nex + 1
        self.ndy = ney + 1
        self.ne = None
        self.nd = None
        # points for domain and boundary
        self.BCs = None

    '''' optimization loop '''

    def optimize(self, opt_pars, den_ft, den_fts, sen_ft, sen_fts, ge_pinn, save_path, case_id):
        if case_id == 'C' or case_id == 'D':
            self.ne = self.nex * self.ney
            self.nd = (self.nex + 1) * (self.ney + 1)
        elif case_id == 'L':
            self.ne = self.nex * self.nex
            self.nd = (self.nex + 1) * (self.nex + 1)
        # optimization parameters
        win_N = opt_pars["win_N"]
        em = opt_pars["em"]
        nu = opt_pars["nu"]
        simp_p = opt_pars["simp_p"]
        vf = opt_pars["vf"]
        move = opt_pars["move"]
        opt_type = opt_pars["opt_type"]
        max_iter_topo = opt_pars["max_iter_topo"]
        t_mat = np.zeros((max_iter_topo, 3))
        obj_cons_gray = np.zeros((max_iter_topo, 4))

        # define design variable
        beta = 0.1
        rho = vf * np.ones(self.ne)  # rho: float64
        rho_tilde = rho.copy()
        eta, rho_phys = heaviside_ft(beta, rho_tilde)

        # define activate/passive elements
        id_active, id_passive = self.find_passive_ele(case_id)
        rho[id_passive] = 0
        rho_phys[id_passive] = 0.

        # set points for domain, boundary and adjoin vector
        self.BCs = SetBCs()
        if case_id == 'C' or case_id == 'D':
            self.BCs.set_connect(self.nex, self.ney)
            self.BCs.set_domain(0., self.length[0], self.nex, 0., self.height[0], self.ney)
        elif case_id == 'L':
            self.BCs.set_connect(self.nex, self.nex)
            self.BCs.set_domain(0., self.length[0], self.nex, 0., self.height[0], self.nex)
        self.BCs.set_load(self.load_size, self.length, self.height, self.load_total, self.ex, case_id)
        self.BCs.set_hard_funcs(self.length[0], self.height[0], case_id)

        # performing optimization
        mma = None
        if opt_type == 'MMA':
            mma = set_mma(rho[id_active], move, 1, 200)  # para: rho, move, m, max_loop

        # initialize
        change, cycle, loop, rho_tol, nd_true = 1., 0, 0, 1.0e-3, None
        req_ele_load = self.find_req_ele()
        ge_pinn.set_flag_nn1(True)
        ge_pinn.set_flag_nn2(False)
        switch_flag = False
        Beta_flag = False
        while change > 1e-4 and loop < max_iter_topo:
            _s0 = time.time()

            # pre-process
            req_ele_rho = np.array(np.where(rho_phys > rho_tol)).squeeze()  # element with rho > rho_tol
            req_ele = np.unique(np.concatenate((req_ele_rho, req_ele_load)))  # all element for training
            req_connect = self.BCs.connect[req_ele, :]
            req_node = np.sort(np.unique(req_connect.flatten()))
            if loop == 0:
                nd_true = len(req_node)
            rate_node = len(req_node) / nd_true
            obj_cons_gray[loop, 3] = rate_node

            # determine the solver of PINN
            obj_cons_gray[loop, 2] = 4 * np.mean(rho_phys[id_active] * (1 - rho_phys[id_active]))
            if obj_cons_gray[loop, 2] <= 0.05:
                switch_flag = True
            if switch_flag:
                ge_pinn.set_flag_nn1(False)
                ge_pinn.set_flag_nn2(True)
                if cycle % win_N == 0:
                    ge_pinn.set_flag_nn1(True)
                    ge_pinn.set_flag_nn2(False)
                    cycle = 0
                if Beta_flag:
                    ge_pinn.set_flag_nn1(True)
                    ge_pinn.set_flag_nn2(False)
                    cycle = 0
                    Beta_flag = False
                cycle = cycle + 1

            # save rho_phys
            path = save_path['end'] + str(loop + 1) + '-rho_phys.csv'
            data = pd.DataFrame(rho_phys)
            data.to_csv(path)

            # get solution
            _s1 = time.time()
            grad_pinn, comp, u_pinn, loss = self.get_sen_obj_dis(ge_pinn, rho_phys, req_ele, req_node, req_connect,
                                                                 em, nu, simp_p)
            _e1 = time.time()
            t_mat[loop, 0] = _e1 - _s1
            print('Training for GE-pinn（' + str(rate_node) + '） took ' + str(t_mat[loop, 0]) + ' s')

            # record compliance
            obj_cons_gray[loop, 0] = comp

            # grad of total element and objective
            df_drp = np.zeros(self.ne)
            df_drp[req_ele] = grad_pinn
            df_drp = (self.ne / obj_cons_gray[
                0, 0]) * df_drp  # scaled with the average compliance of the first iteration

            # sensitivity filter
            df_drp = np.array((sen_ft @ (rho[np.newaxis].T * df_drp[np.newaxis].T)) / sen_fts)
            df_drp = df_drp / np.maximum(1.0e-3, rho[np.newaxis].T)

            # density or heaviside filter
            dv_drp = np.ones_like(df_drp)  # scaled with the total number of element, in fact: dv_drp / self.ne
            # step 1
            drp_drt = beta * (1 - np.tanh(beta * (rho_tilde[np.newaxis].T - eta)) ** 2) / (
                        np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
            # step 2
            df_dr = np.array(den_ft @ ((df_drp * drp_drt) / den_fts))
            dv_dr = np.array(den_ft @ ((dv_drp * drp_drt) / den_fts))
            del df_drp, dv_drp, drp_drt

            # plot result
            # path = save_path['dis'] + str(loop + 1) + '_Disp.png'
            # plot_dis(u_pinn, req_node, self.BCs.dom, path, 5 * self.ex)

            # update variable with scaled constrain
            _s2 = time.time()
            if opt_type == 'MMA':
                v_val = self.ne * (np.mean(rho_phys[id_active]) - vf)
                rho[id_active] = topo_mma(rho[id_active], comp, df_dr[id_active], v_val, dv_dr[id_active], loop + 1)
            if opt_type == 'OC':
                rho = topo_oc(rho, rho_phys, df_dr, dv_dr, vf, move, id_active, id_passive)

            rho[id_passive] = 0.
            _e2 = time.time()
            t_mat[loop, 1] = _e2 - _s2

            if loop+1 >= 6:
                obj_old = obj_cons_gray[loop - 5:loop - 2, 0]
                obj_new = obj_cons_gray[loop - 2:loop+1, 0]
                change = sum(abs(obj_old - obj_new)) / sum(obj_new)

            # post progress
            if loop % 5 == 4 and beta < 24.:
                beta = np.minimum(24., beta * 2.)
                Beta_flag = True
                print('2X increase in beta: ' + str(beta))

            # filter design variable
            rho_tilde = np.array((den_ft @ rho[np.newaxis].T) / den_fts)
            rho_tilde[id_passive] = 0.
            eta, rho_phys = heaviside_ft(beta, rho_tilde)
            rho_phys[id_passive] = 0.
            # path = save_path['den'] + str(loop + 1) + '_Den.png'
            # if case_id == 'C' or case_id == 'D':
            #     plot_sen_den(rho_phys, self.nex, self.length[0], self.ney, self.height[0], 'den', path)
            # elif case_id == 'L':
            #     plot_sen_den(rho_phys, self.nex, self.length[0], self.nex, self.length[0], 'den', path)

            rho = rho.flatten()
            rho_tilde = rho_tilde.flatten()
            rho_phys = rho_phys.flatten()
            obj_cons_gray[loop, 1] = np.sum(rho_phys) / len(id_active)
            # complete the iteration and print result
            _e0 = time.time()
            t_mat[loop, 2] = _e0 - _s0
            print('Performing current cycle took ' + str(t_mat[loop, 2]) + ' s')
            v1 = obj_cons_gray[loop, 0]  # compliance
            v2 = obj_cons_gray[loop, 1]  # volume
            v3 = obj_cons_gray[loop, 2]  # gray
            loop = loop + 1
            print("it.:{0} || obj.:{1:.4f} || Vol.:{2:.4f}/({3:.4f}) || "
                  "ch.:{4:.4f} || gray.:{5:.4f} \n".format(loop, v1, v2, vf, change, v3))

            # # save point and displacement
            # path = save_path['end'] + str(loop) + '-point.csv'
            # point_dom = self.BCs.dom[req_node, :]
            # data = pd.DataFrame(point_dom)
            # data.to_csv(path)
            # path = save_path['end'] + str(loop) + '-dis.csv'
            # data = pd.DataFrame(u_pinn)
            # data.to_csv(path)

        # plot result
        path = save_path['den'] + 'end_Den.png'
        if case_id == 'C' or case_id == 'D':
            plot_sen_den(rho_phys, self.nex, self.length[0], self.ney, self.height[0], 'den', path)
        elif case_id == 'L':
            plot_sen_den(rho_phys, self.nex, self.length[0], self.nex, self.length[0], 'den', path)

        # save point and displacement
        path = save_path['dis'] + str(loop) + '-point.csv'
        point_dom = self.BCs.dom[req_node, :]
        data = pd.DataFrame(point_dom)
        data.to_csv(path)
        path = save_path['dis'] + str(loop) + '-dis.csv'
        data = pd.DataFrame(u_pinn)
        data.to_csv(path)

        # save rho_phys
        path = save_path['end'] + 'end-rho_phys.csv'
        data = pd.DataFrame(rho_phys)
        data.to_csv(path)

        # plot iterative history
        path = [save_path['end'] + 'opt_train_time.png', save_path['end'] + 'opt_compliance.png']
        plot_time_obj(t_mat, obj_cons_gray, path)

        # save time; and gray;
        path = save_path['end'] + 'time.csv'
        data = pd.DataFrame(t_mat)
        data.to_csv(path)

        # save histories of objective, constrain and gray
        path = save_path['end'] + 'obj_cons_gray.csv'
        data = pd.DataFrame(obj_cons_gray)
        data.to_csv(path)

    ''' find passive elements '''

    def find_passive_ele(self, case_id):
        id_active, id_passive = None, None
        if case_id == 'L':
            id_ele = np.reshape(np.array(np.arange(self.ne)), [self.nex, self.nex], order='F')
            id_passive = id_ele[(self.nex - self.ney + 1):, :][:, (self.nex - self.ney + 1):]
            id_passive = np.reshape(id_passive, [id_passive.size, 1], order='F')
            id_ele = np.reshape(id_ele, [self.ne, 1], order='F')
            id_active = np.setdiff1d(id_ele, id_passive)
            id_passive = id_passive.squeeze()
        elif case_id == 'C' or case_id == 'D':
            id_active = np.array(np.arange(self.ne))
            id_passive = []
        return id_active, id_passive

    ''' find elements associated with the load nodes '''

    def find_req_ele(self):
        idx = self.BCs.load_vec['idx']
        id_load_ele = []
        for _, value in enumerate(idx):
            id_temp = np.where(self.BCs.connect == value)
            for _, id_ in enumerate(id_temp[0]):
                id_load_ele.append(id_)
        id_load_ele = np.unique(np.array(id_load_ele).flatten())
        return id_load_ele

    ''' get sensitivity, compliance, and displacement '''

    def get_sen_obj_dis(self, pinn_model, rho_phys, req_ele, req_node, req_connect, em, nu, simp_p):
        ele_geo = [self.ex, self.ey]
        # build mapping
        node_map = np.zeros((self.nd, 2), dtype=int)
        id_ = 0
        for _, value_ in enumerate(req_node):
            node_map[value_, 0] = value_
            node_map[value_, 1] = id_
            id_ = id_ + 1
        id_N1 = node_map[req_connect[:, 0], 1][np.newaxis].T
        id_N2 = node_map[req_connect[:, 1], 1][np.newaxis].T
        id_N3 = node_map[req_connect[:, 2], 1][np.newaxis].T
        id_N4 = node_map[req_connect[:, 3], 1][np.newaxis].T
        id_node = np.concatenate((id_N1, id_N2, id_N3, id_N4), axis=1)
        # required point
        req_dom = self.BCs.dom[req_node, :]
        req_rho = rho_phys[req_ele]
        id_load = node_map[self.BCs.load_vec['idx'], 1]
        # training model and predicting displacement
        df_drp, comp, u, loss = pinn_model.training(req_dom, id_node, req_rho, em, nu, simp_p, ele_geo,
                                                    self.BCs.load_vec, id_load, self.BCs.func_x, self.BCs.func_y)
        # taking variables out of the calculation graph
        df_drp_np = df_drp.detach().cpu().numpy()
        comp_np = comp.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        return df_drp_np, comp_np, u_np, loss
