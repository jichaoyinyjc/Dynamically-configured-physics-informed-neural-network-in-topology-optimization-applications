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

'C': cantilever beam       data_C
'D': double clamped beam   data_D
'L': L-shape beam          data_L
'''


def get_path(str_):
    data_ = None
    path_ = None
    type_ = None
    if str_ == 'C_low':
        type_ = "C"
        data_ = cfg.data_C_low
        path_ = {"dis": './T_D_1/C_low/dis/', "den": './T_D_1/C_low/den/', "end": './T_D_1/C_low/end/'}
    elif str_ == 'C_high':
        type_ = "C"
        data_ = cfg.data_C_high
        path_ = {"dis": './T_D_1/C_high/dis/', "den": './T_D_1/C_high/den/', "end": './T_D_1/C_high/end/'}
    elif str_ == 'L_low':
        type_ = "L"
        data_ = cfg.data_L_low
        path_ = {"dis": './T_D_1/L_low/dis/', "den": './T_D_1/L_low/den/', "end": './T_D_1/L_low/end/'}
    elif str_ == 'L_high':
        type_ = "L"
        data_ = cfg.data_L_high
        path_ = {"dis": './T_D_1/L_high/dis/', "den": './T_D_1/L_high/den/', "end": './T_D_1/L_high/end/'}
    return data_, path_, type_


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
    case_idx_list = ['C_low', 'L_low', 'C_high', 'L_high']  # case 'C_low' has been tested in 'tol'
    rad_list = [2., 2., 5., 5.]
    test_id = 1

    #
    case_idx = case_idx_list[test_id]
    config_data, save_path, case_type = get_path(case_idx)
    #
    length, height, nex, ney, load_size, load_total = get_data(config_data)
    # generate path
    try:
        os.makedirs(save_path['dis'])
        os.makedirs(save_path['den'])
        os.makedirs(save_path['end'])
    except:
        print('error in generating path')
    #
    ex = length[0] / nex  # length of element edge
    rad_d = rad_list[test_id]  # density filter radius
    rad_s = rad_list[test_id]  # sensitivity filter radius
    #
    ''' initialize density/sensitivity filter model '''
    t_filter, den_ft, den_fts, sen_ft, sen_fts = None, None, None, None, None
    if case_type == 'C':
        _s1 = time.time()
        den_ft, den_fts = get_ft(nex, ney, rad_d)
        sen_ft, sen_fts = get_ft(nex, ney, rad_s)
        _e1 = time.time()
        t_filter = _e1 - _s1
    elif case_type == 'L':
        _s1 = time.time()
        den_ft, den_fts = get_ft(nex, nex, rad_d)
        sen_ft, sen_fts = get_ft(nex, nex, rad_s)
        _e1 = time.time()
        t_filter = _e1 - _s1
    print('building the filters took ' + str(t_filter) + ' s, and the radii are ' +
          str(rad_s) + ' and ' + str(rad_d) + ', respectively.\n')

    ''' initialize PINN model '''
    ge_pinn = Pinn(cfg.net_pars, cfg.net_2_pars, cfg.train_pars)

    ''' initialize topology optimization model '''
    topo_model = Topo(length, height, nex, ney, load_total, load_size)

    '''optimization cycle'''
    topo_model.optimize(cfg.opt_pars, den_ft, den_fts, sen_ft, sen_fts, ge_pinn, save_path, case_type)
