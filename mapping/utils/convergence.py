import numpy as np


def check_array_convergence(last_x_arr, last_y_arr, curr_x_arr, curr_y_arr, stop_abs, stop_rel, threshold=1e-15):
    """
        Check the convergence conditions |last_i_arr - curr_i_arr|/|curr_i_arr| < stop_rel
        and  |last_i_arr - curr_i_arr| < stop_abs
        for all entries in the arrays individually and for i = x, y
        Threshold marks the value in both the curr_i-arrays below which the relative error is no longer applied
        due to the danger of division by 0
    :return: Bool, which states whether or not convergence was successful or not
    """
    if len(curr_x_arr):
        for l_x, c_x in zip(last_x_arr, curr_x_arr):
            abs_diff = np.abs(c_x - l_x)
            if c_x < threshold:
                if not abs_diff <= stop_abs:
                    return False
            else:
                if not (abs_diff <= stop_abs and abs_diff <= stop_rel * max(l_x, c_x)):
                    return False
    if len(curr_y_arr) > 0:
        for l_y, c_y in zip(last_y_arr, curr_y_arr):
            abs_diff = np.abs(c_y - l_y)
            if c_y < threshold:
                if not abs_diff <= stop_abs:
                    return False
            else:
                if not (abs_diff <= stop_abs and abs_diff <= stop_rel * max(l_y, c_y)):
                    return False
    return True


def check_convergence(last_x, last_y, curr_x, curr_y, stop_abs, stop_rel, threshold=1e-15):
    """
        Check the convergence conditions |last_i - curr_i|/|curr_i| < stop_rel
        and  |last_i - curr_i| < stop_abs
        for i = x, y
        Threshold marks the value in both the curr_i-arrays below which the relative error is no longer applied
        due to the danger of division by 0
    :return: Bool, which states whether or not convergence was successful or not
    """
    abs_diff_alpha = np.abs(curr_x - last_x)
    if curr_x < threshold:
        check_alpha = abs_diff_alpha <= stop_abs
    else:
        check_alpha = abs_diff_alpha <= stop_abs and \
                      abs_diff_alpha <= stop_rel * max(np.abs(last_x), np.abs(curr_x))
    abs_diff_beta = np.abs(curr_y - last_y)
    if curr_y < threshold:
        check_beta = abs_diff_beta <= stop_abs
    else:
        check_beta = abs_diff_beta <= stop_abs and \
                     abs_diff_beta <= stop_rel * max(np.abs(last_y), np.abs(curr_y))
    return check_alpha and check_beta