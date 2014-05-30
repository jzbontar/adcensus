extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"
#include <math_constants.h>

#define TB 128

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

__global__ void ad(float *x0, float *x1, float *output, int size, int size3, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;
		int xy = d % size23;
		d /= size23;

		float dist = 0;
		if (x - d < 0) {
			dist = CUDART_INF;
		} else {
			for (int i = 0; i < 3; i++) {
				int ind = i * size23 + xy;
				dist += fabsf(x0[ind] - x1[ind - d]);
			}
		}
		output[id] = dist / 3;
	}
}

int ad(lua_State *L)
{
	THCudaTensor *x0 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	ad<<<(THCudaTensor_nElement(output) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(x0),
		THCudaTensor_data(x1),
		THCudaTensor_data(output),
		THCudaTensor_nElement(output),
		THCudaTensor_size(output, 3),
		THCudaTensor_size(output, 2) * THCudaTensor_size(output, 3));
	checkCudaError(L);
	return 0;
}


__global__ void census(float *x0, float *x1, float *output, int size, int size2, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;
		d /= size3;
		int y = d % size2;
		d /= size2;

		float dist;
		if (x - d < 0) {
			dist = CUDART_INF;
		} else {
			dist = 0;
			for (int i = 0; i < 3; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - 3; yy <= y + 3; yy++) {
					for (int xx = x - 4; xx <= x + 4; xx++) {
						if (0 <= xx - d && xx < size3 && 0 <= yy && yy < size2) {
							int ind_q = (i * size2 + yy) * size3 + xx;
							if ((x0[ind_q] < x0[ind_p]) != (x1[ind_q - d] < x1[ind_p - d])) {
								dist++;
							}
						} else {
							dist++;
						}
					}
				}
			}
		}
		output[id] = dist / 3;
	}
}


int census(lua_State *L)
{
	THCudaTensor *x0 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	census<<<(THCudaTensor_nElement(output) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(x0),
		THCudaTensor_data(x1),
		THCudaTensor_data(output),
		THCudaTensor_nElement(output),
		THCudaTensor_size(output, 2),
		THCudaTensor_size(output, 3));
	checkCudaError(L);
	return 0;
}

__global__ void spatial_argmin(float *input, float *output, int size, int size1, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		int argmin = 0;
		float min = 2e38;
		for (int i = 0; i < size1; i++) {
			float val = input[(dim0 * size1 + i) * size23 + dim23];
			if (val < min) {
				min = val;
				argmin = i;
			}
		}
		output[id] = argmin + 1;
	}
}

int spatial_argmin(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

	spatial_argmin<<<(THCudaTensor_nElement(output) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(input),
		THCudaTensor_data(output),
		THCudaTensor_nElement(output),
		THCudaTensor_size(input, 1),
		THCudaTensor_size(input, 2) * THCudaTensor_size(output, 3));
	checkCudaError(L);
	return 0;
}

#define CBCA_CONDITIONS(xx, yy, xxx, yyy) (\
    0 <= (yy) && (yy) < size2 && \
    0 <= (xx) && (xx) < size3 && \
    fabsf(x0[(yy) * size3 + (xx)] - x0[(yyy) * size3 + (xxx)]) < tau1 && ((\
		fabsf((xx) - x) + fabsf((yy) - y) < L1 && \
		fabsf(x0[(yy) * size3 + (xx)] - x0[y * size3 + x]) < tau2 && \
		fabsf(x1[(yy) * size3 + (xx) - d] - x1[y * size3 + x - d]) < tau2) || (\
		fabsf((xx) - x) + fabsf((yy) - y) < L2 && \
		fabsf(x0[(yy) * size3 + (xx)] - x0[y * size3 + x]) < tau1 && \
		fabsf(x1[(yy) * size3 + (xx) - d] - x1[y * size3 + x - d]) < tau1)))

__global__ void cbca(float *x0, float *x1, float *vol, float *vol_out, int size, int size2, int size3, int L1, int L2, float tau1, float tau2)
{
    int output_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_id < size) {
		int d = output_id;
        const int x = d % size3;
        d /= size3;
        const int y = d % size2;
        d /= size2;

        int yn, ys;
        for (yn = y - 1; CBCA_CONDITIONS(x, yn, x, yn + 1); yn--) {};
        for (ys = y + 1; CBCA_CONDITIONS(x, ys, x, ys - 1); ys++) {};

        float sum = 0;
        int cnt = 0;

        /* output */
        for (int yy = yn + 1; yy < ys; yy++) {
            int xe, xw;
            for (xe = x - 1; CBCA_CONDITIONS(xe, yy, xe + 1, yy); xe--) {};
            for (xw = x + 1; CBCA_CONDITIONS(xw, yy, xw - 1, yy); xw++) {};

            for (int xx = xe + 1; xx < xw; xx++) {
				sum += vol[(d * size2 + yy) * size3 + xx];
				cnt++;
            }
        }
		vol_out[output_id] = sum / cnt;
	}
}

/* cross-based cost aggregation */
int cbca(lua_State *L)
{
    THCudaTensor *x0 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
    THCudaTensor *x1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
    THCudaTensor *vol = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *vol_out = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
    int L1 = luaL_checkinteger(L, 5);
    int L2 = luaL_checkinteger(L, 6);
    float tau1 = luaL_checknumber(L, 7);
    float tau2 = luaL_checknumber(L, 8);

    cbca<<<(THCudaTensor_nElement(vol) - 1) / TB + 1, TB>>>(
        THCudaTensor_data(x0),
        THCudaTensor_data(x1),
        THCudaTensor_data(vol),
        THCudaTensor_data(vol_out),
        THCudaTensor_nElement(vol),
        THCudaTensor_size(x0, 2),
        THCudaTensor_size(x0, 3),
        L1, L2, tau1, tau2);
    checkCudaError(L);
    return 0;
}

/* median 3x3 filter */
__global__ void median3(float *img, float *out, int size, int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		const int x = id % width;
		const int y = id / width;

		float w[9] = {
			y == 0          || x == 0         ? 0 : img[id - width - 1],
			y == 0                            ? 0 : img[id - width],
			y == 0          || x == width - 1 ? 0 : img[id - width + 1],
			                   x == 0         ? 0 : img[id         - 1],
			                                        img[id],
			                   x == width - 1 ? 0 : img[id         + 1],
			y == height - 1 || x == 0         ? 0 : img[id + width - 1],
			y == height - 1                   ? 0 : img[id + width],
			y == height - 1 || x == width - 1 ? 0 : img[id + width + 1]
		};

		for (int i = 0; i < 5; i++) {
			float tmp = w[i];
			int idx = i;
			for (int j = i + 1; j < 9; j++) {
				if (w[j] < tmp) {
					idx = j;
					tmp = w[j];
				}
			}
			w[idx] = w[i];
			w[i] = tmp;
		}

		out[id] = w[4];
	}
}

int median3(lua_State *L)
{
	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *out = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

	median3<<<(THCudaTensor_nElement(img) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(img),
		THCudaTensor_data(out),
		THCudaTensor_nElement(img),
		THCudaTensor_size(img, 2),
		THCudaTensor_size(img, 3));

	checkCudaError(L);
	return 0;
}

static const struct luaL_Reg funcs[] = {
	{"ad", ad},
	{"median3", median3},
	{"census", census},
	{"cbca", cbca},
	{"spatial_argmin", spatial_argmin},
	{NULL, NULL}
};

extern "C" int luaopen_libadcensus(lua_State *L) {
	luaL_openlib(L, "adcensus", funcs, 0);
	return 1;
}
