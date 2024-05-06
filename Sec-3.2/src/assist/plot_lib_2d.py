import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

''' plot displacement '''


def plot_dis(u, req_node, coord, save_path, s):
    u_x = u[:, 0]
    u_y = u[:, 1]
    X = np.expand_dims(coord[req_node, 0] + u_x, axis=1)
    Y = np.expand_dims(coord[req_node, 1] + u_y, axis=1)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16., 6.))
    # subplot 1
    cfig1 = axs[0].scatter(X, Y, c=u_x, s=s, vmin=u_x.min(), vmax=u_x.max(), cmap='coolwarm')
    axes_1 = inset_axes(
        axs[0],
        width=0.15,  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=axs[0].transAxes,
        borderpad=0,
    )
    axs[0].set_aspect('equal')
    axs[0].set_title('displacement X')
    plt.colorbar(cfig1, cax=axes_1)
    # subplot 2
    cfig2 = axs[1].scatter(X, Y, c=u_y, s=s, marker='o', vmin=u_y.min(), vmax=u_y.max(), cmap='coolwarm')
    axes_2 = inset_axes(
        axs[1],
        width=0.15,  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=axs[1].transAxes,
        borderpad=0,
    )
    axs[1].set_aspect('equal')
    axs[1].set_title('displacement Y')
    plt.colorbar(cfig2, cax=axes_2)
    # save
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


''' plot sensitivity or density'''


def plot_sen_den(vec, nex, length, ney, height, flag, save_path):
    vec = vec.reshape([ney, nex], order='F')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    if flag == 'sen':
        cfig = axs.imshow(np.flipud(vec), extent=[0., length, 0., height], cmap='jet')
        title_ = 'sensitivity'
        axs.set_aspect('equal')
        axs.set_title(title_)
        axes_ = inset_axes(
            axs,
            width="5%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=axs.transAxes,
            borderpad=0,
        )
        cb = plt.colorbar(cfig, cax=axes_)
        det_vec = np.max(vec) - np.min(vec)
        cb.set_ticks(np.min(vec) + [0, 0.25 * det_vec, 0.5 * det_vec, 0.75 * det_vec, det_vec])
    elif flag == 'den':
        axs.imshow(np.flipud(vec), vmin=0., vmax=1., extent=[0., length, 0., height], cmap='binary')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


''' plot time and objective '''


def plot_time_obj(time_mat, obj_con_gray, save_path):
    time_cost = time_mat[:, -1]
    plt.figure()
    plt.plot(np.arange(len(time_cost)) + 1, time_cost)
    plt.xlim((1, len(time_cost)))
    plt.ylim((0, np.ceil(time_cost.max())))
    plt.xlabel('TopOpt iteration')
    plt.ylabel('DEM train time [s]')
    plt.savefig(save_path[0], dpi=300, bbox_inches="tight")
    plt.close()

    obj_history = obj_con_gray[:, 0]
    plt.figure()
    plt.plot(np.arange(len(obj_history)) + 1, obj_history)
    plt.xlim((1, len(obj_history)))
    plt.xlabel('TopOpt iteration')
    plt.ylabel('Compliance [J]')
    plt.savefig(save_path[1], dpi=300, bbox_inches="tight")
    plt.close()
