import numpy as np


def z_score(x, mean, std):
    return (x - mean) / std


def z_inverse(x, mean, std):
    return x * std + mean


def MAPE(v, v_):
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    dim = len(y_.shape)

    if dim == 3:
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        tmp_list = []
        y = np.swapaxes(y, 0, 1)
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
