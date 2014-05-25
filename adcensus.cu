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
				dist += fabs(x0[ind] - x1[ind - d]);
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

		float dist = 0;
		if (x - d < 0) {
			dist = CUDART_INF;
		} else {
			int cnt = 0;
			for (int i = 0; i < 3; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - 4; yy <= y + 4; yy++) {
					for (int xx = x - 4; xx <= x + 4; xx++) {
						if (0 <= xx - d && xx < size3 && 0 <= yy && yy < size2) {
							int ind_q = (i * size2 + yy) * size3 + xx;
							if ((x0[ind_p] - x0[ind_q]) * (x1[ind_p - d] - x1[ind_q - d]) < 0) {
								dist++;
							}
							cnt++;
						}
					}
				}
			}
			assert(cnt > 0);
			dist /= cnt;
		}
		output[id] = dist;
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


static const struct luaL_Reg funcs[] = {
	{"ad", ad},
	{"census", census},
	{"spatial_argmin", spatial_argmin},
	{NULL, NULL}
};

extern "C" int luaopen_libadcensus(lua_State *L) {
	luaL_openlib(L, "adcensus", funcs, 0);
	return 1;
}
