import numpy as np
import random

DEBUG = False
VERBOSE_LEVEL = 0
KNOWN_REWARDS = False
KNOWN_KERNELS = False

def unimplemented(*args):
    msg = "[Unimplemented]:" + " ".join(str(arg) for arg in args)
    if DEBUG:
        raise Exception(msg)
    else:
        print(msg)

def inaccessible(*args):
    msg = "[FATAL]: Inaccessible" + " ".join(str(arg) for arg in args)
    raise Exception(msg)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def time_str(sec):
    s = int(sec)
    m = s // 60
    h = m // 60
    m = m % 60
    s = s % 60
    if h: 
        return f"{h}h{m:2d}m{s:2d}s"
    if m:
        return f"{m}m{s:2d}s"
    return f"{s}s"

def _reduce_curve_linear(curve, num=20):
    size = len(curve)
    dp   = (size - 1) / (num - 1)
    p    = 0
    simpl = []
    while p < size - 1:
        p_int = int(p)
        rem   = p - p_int
        point = (1.0 - rem) * curve[p_int] + rem * curve[p_int+1]
        p    += dp
        simpl.append(point)
    simpl.append(curve[-1])
    return simpl

def _reduce_curve_log(curve, num=20):
    size  = len(curve)
    dp    = np.log(size) / (num - 1)
    simpl = []
    for i in range(num):
        point = curve[int(np.exp(i * dp))-1]
        simpl.append(point)
    return simpl

def reduce_curve(curve, num=20, mode="linear"):
    """ 
    To reduce a curve. Reduce the curve to the specified number of points.
    Two possible reduction methods:

    - linear: points are spread linearly, for standard plots;
    - log   : points are spread with geometric stepsizes, for logscale plots.
    """

    if   mode == "linear": return _reduce_curve_linear(curve, num)
    elif mode == "log": return _reduce_curve_log(curve, num)
    else: return curve

def smooth_cloud(cloud, window=100, l=100):
    """
    Given a cloud of points, as a collection of elements (x, y), compute a curve
    c such that c(x) is the average of y-coordinates of elements with 
    approximately the same x-coordinate than x. 
    - window: parametrize how large the averaging window is. 
    """
    cloud = sorted(cloud)
    min_x, max_x = cloud[0][0], cloud[-1][0]
    cur_n = 0
    cur_s = 0
    i, j  = 0, 0
    curve = []
    for x in range(0, max_x):
        while j < len(cloud) and cloud[j][0] < x + 0.5*window:
            cur_s += cloud[j][1]
            cur_n += 1
            j += 1
        while i < len(cloud) and cloud[i][0] < x - 0.5*window:
            cur_s -= cloud[i][1]
            cur_n -= 1
            i += 1
        curve.append(cur_s/max(1.0, cur_n))
    # if len(curve) > l: curve = curve[:l]
    # if len(curve) < l: curve = curve + [curve[-1] for _ in range(l-len(curve))]
    return curve
