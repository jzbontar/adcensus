extern "C" {
    #include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

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

		float dist;
		if (x - d < 0) {
			dist = CUDART_NAN;
		} else {
			for (int i = 0; i < 3; i++) {
				int ind = i * size23 + xy;
				dist += fabsf(x0[ind] - x1[ind - d]) / 3.;
			}
		}
		output[id] = dist;
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

static const struct luaL_Reg funcs[] = {
	{"ad", ad},
	{NULL, NULL}
};

extern "C" int luaopen_libadcensus(lua_State *L) {
    luaL_openlib(L, "adcensus", funcs, 0);
    return 1;
}
