import numpy as np
cimport numpy as np

cdef int height = 288 
cdef int width = 384 
cdef int disp_max = 16 

def census(np.ndarray[np.float32_t, ndim=3] x0, np.ndarray[np.float32_t, ndim=3] x1):
    cdef np.ndarray[np.float32_t, ndim=3] vol = np.empty((disp_max, height, width), dtype=np.float32)

    cdef int d, i, j, k, ii, jj, cnt
    cdef np.float32_t v1, v2, dist

    for d in range(disp_max):
        for i in range(height):
            for j in range(width):
                if j - d < 0:
                    vol[d, i, j] = np.inf
                    continue

                dist = 0
                cnt = 0
                for k in range(3):
                    for ii in range(i - 4, i + 5):
                        for jj in range(j - 4, j + 5):
                            if 0 <= ii < height and 0 <= jj < width and 0 <= jj - d < width:
                                cnt += 1
                                v1 = (x0[ii, jj, k] - x0[i, j, k])
                                v2 = (x1[ii, jj - d, k] - x1[i, j - d, k])
                                if v1 * v2 < 0:
                                    dist += 1
                vol[d, i, j] = dist / cnt
    return vol
