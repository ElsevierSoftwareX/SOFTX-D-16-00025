/*
   GPU code for analytic continuation through a sampling method
   Copyright (C) 2016  Johan Nordström

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef __KERNELS_CUH
#define __KERNELS_CUH

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/complex.h>
#include "constants.cuh"

namespace mish{

	typedef thrust::complex<float> (*kernelFunction)(float,float);

	//Kernel 0 should not be modified since the analytic integral is hardcoded
	__device__ thrust::complex<float> kernel0(float x, float omega)
	{
		thrust::complex<float> i(0.0f, 1.0);
		return 1.0f/(i*x - omega);
	}
	
	__device__ thrust::complex<float> kernel1(float x, float omega)
	{
		return 1 * __expf(-x * omega) / (1 + __expf(-D_BETA * omega));
	}

	__device__ thrust::complex<float> kernel2(float x, float omega)
	{
		return omega*omega/(x*x + omega*omega);
	}

	//All kernels should be added here
	__constant__ kernelFunction kernelList[3] = {kernel0, kernel1, kernel2};
}


#endif
