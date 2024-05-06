import os
import time
import src.assist.config as cfg
from src.opt.topo_2d import Topo
from src.net.pinn_2d import Pinn
from src.pre.filter_2d import get_ft

'''
Length unit    :   m
Load unit      :   N
Pressure unit  :   Pa
'''


def get_path(vol):
    data_ = cfg.data_geo
    if vol == 0.3:
        path_ = {"dis": './T_D_0.3/dis/',
                 "den": './T_D_0.3/den/',
                 "end": './T_D_0.3/end/'}
    elif vol == 0.5:
        path_ = {"dis": './T_D_0.5/dis/',
                 "den": './T_D_0.5/den/',
                 "end": './T_D_0.5/end/'}
    return data_, path_

def get_data(cfg_data):
    length_ = cfg_data["length"]
    height_ = cfg_data["height"]  # shape of design domain
    nex_ = cfg_data["nex"]
    ney_ = cfg_data["ney"]  # discrete element
    load_size_ = cfg_data["load_size"]  # length of load boundary
    load_total_ = cfg_data["load_total"]  # magnitude of the combined force
    return length_, height_, nex_, ney_, load_size_, load_total_


# main code
if __name__ == '__main__':
    #
    config_data, save_path = get_path(cfg.opt_pars["vf"])
    #
    length, height, nex, ney, load_size, load_total = get_data(config_data)
    #
    try:
        os.makedirs(save_path['dis'])
        os.makedirs(save_path['den'])
        os.makedirs(save_path['end'])
    except:
        print('error in generating path')
    #
    ex = length[0] / nex  # length of element edge
    rad_d = 2.67  # density filter radius
    rad_s = 2.67  # sensitivity filter radius
    #
    ''' initialize density/sensitivity filter model '''
    _s1 = time.time()
    den_ft, den_fts = get_ft(nex, ney, rad_d)
    sen_ft, sen_fts = get_ft(nex, ney, rad_s)
    _e1 = time.time()
    t_filter = _e1 - _s1
    print('building the filters took ' + str(t_filter) + ' s, and the radii are ' +
          str(rad_s) + ' and ' + str(rad_d) + ', respectively.\n')

    ''' initialize PINN model '''
    ge_pinn_a = Pinn(cfg.net_pars, cfg.net_2_pars, cfg.train_pars)
    ge_pinn_b = Pinn(cfg.net_pars, cfg.net_2_pars, cfg.train_pars)

    ''' initialize topology optimization model '''
    topo_model = Topo(length, height, nex, ney, load_total, load_size)

    '''optimization cycle'''
    topo_model.optimize(cfg.opt_pars, den_ft, den_fts, sen_ft, sen_fts, ge_pinn_a, ge_pinn_b, save_path)

