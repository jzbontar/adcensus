#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

cdef int height = 288 
cdef int width = 384 
cdef int disp_max = 16 

def census(np.ndarray[np.float64_t, ndim=3] x0, np.ndarray[np.float64_t, ndim=3] x1):
    cdef np.ndarray[np.float64_t, ndim=3] vol
    cdef int d, i, j, k, ii, jj, cnt, v1, v2
    cdef np.float64_t dist

    vol = np.empty((disp_max, height, width), dtype=np.float64)

    for d in range(disp_max):
        for i in range(height):
            for j in range(width):
                if j - d < 0:
                    vol[d, i, j] = np.inf
                    continue

                dist = 0
                for k in range(3):
                    for ii in range(i - 3, i + 4):
                        for jj in range(j - 4, j + 5):
                            if 0 <= ii < height and 0 <= jj < width and 0 <= jj - d < width:
                                v1 = x0[ii, jj, k] < x0[i, j, k]
                                v2 = x1[ii, jj - d, k] < x1[i, j - d, k]
                                if v1 != v2:
                                    dist += 1
                            else:
                                dist += 1
                vol[d, i, j] = dist / 3
    return vol

def census_transform(np.ndarray[np.float64_t, ndim=3] x):
    cdef np.ndarray[np.int_t, ndim=3] cen
    cdef int i, j, ii, jj, k, ind, ne

    ne = np.random.randint(2**31)
    cen = np.zeros((height, width, 63 * 3), dtype=np.int)
    for i in range(height):
        for j in range(width):
            ind = 0
            for ii in range(i - 3, i + 4):
                for jj in range(j - 4, j + 5):
                    if 0 <= ii < height and 0 <= jj < width:
                        for k in range(3):
                            cen[i, j, ind] = x[ii, jj, k] < x[i, j, k]
                            ind += 1
                    else:
                        cen[i, j, ind + 0] = ne
                        cen[i, j, ind + 1] = ne
                        cen[i, j, ind + 2] = ne
                        ind += 3
            assert(ind == 63 * 3)
    return cen
