import numpy as np


class SetBCs:
    def __init__(self):
        super(SetBCs, self).__init__()
        self.dom = None
        self.load_vec = None
        self.func_x = None
        self.func_y = None
        self.connect = None

    ''' set point for domain '''

    def set_domain(self, min_x, length, nex, min_y, height, ney):
        ndx = nex + 1
        ndy = ney + 1
        nd = ndx * ndy
        lin_x = np.linspace(min_x, length, ndx)
        lin_y = np.linspace(min_y, height, ndy)
        # set coordinates of domain points
        self.dom = np.zeros((nd, 2))
        id_ = 0
        for coord_x in np.nditer(lin_x):
            tb = ndy * id_
            id_ += 1
            te = tb + ndy
            self.dom[tb:te, 0] = coord_x
            self.dom[tb:te, 1] = lin_y

    ''' set point for load '''

    def set_load(self, load_size, length, height, load_total, ex, case_id):
        idx = None
        if case_id == 'C':
            idx = np.array(np.where((self.dom[:, 0] >= (length[0] - load_size))
                                    & (self.dom[:, 1] == height[0]))).squeeze()
        elif case_id == 'L':
            idx = np.array(np.where((self.dom[:, 0] >= (length[0] - load_size))
                                    & (self.dom[:, 1] == (length[0] - length[1]) / 2.))).squeeze()
        pts = self.dom[idx, :]
        value = np.ones(np.shape(pts)) * [0., load_total / len(pts)]
        self.load_vec = {'idx': idx, 'coord': pts, 'value': value}

    ''' set hard constrain functions '''

    def set_hard_funcs(self, length, height, case_id):
        if case_id == 'C':
            self.func_x = lambda x: x[:, 0] / length
            self.func_y = lambda x: x[:, 0] / length
        elif case_id == 'L':
            self.func_x = lambda x: 1.0 - x[:, 1] / height
            self.func_y = lambda x: 1.0 - x[:, 1] / height

    ''' set connect '''

    def set_connect(self, nex, ney):
        ndx, ndy = nex + 1, ney + 1
        nd = ndx * ndy
        node_set = np.array(np.split(np.arange(nd), ndx)).T
        node_set_1 = node_set[0:ney, 0:nex]  # upper left
        node_set_2 = node_set[1:ndy, 0:nex]  # lower left
        node_set_3 = node_set[1:ndy, 1:ndx]  # lower right
        node_set_4 = node_set[0:ney, 1:ndx]  # upper right
        self.connect = np.concatenate(
            (node_set_1.T.flatten()[np.newaxis].T, node_set_2.T.flatten()[np.newaxis].T,
             node_set_3.T.flatten()[np.newaxis].T, node_set_4.T.flatten()[np.newaxis].T), axis=1)
