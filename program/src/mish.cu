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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/complex.h>
#include "config.cuh"
#include "kernels.cuh"
#include "constants.cuh"
#include "mish.cuh"


//Error checking macro for the CUDA kernels
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace mish 
{
	
	//Device buffers, all arrays are of the same length
	//as the number of global updates
	struct devicePointers{
		configuration *min_conf, *cur_conf;
		complexFunction greens, *cur_greens, *eps_greens;
		complexFunction *epsh_greens, *opt_greens, *tmp_greens;
		float* min_dev, *cur_dev, *eps_dev, *epsh_dev, *opt_dev;
		float* eps, *epsh, *opt_eps;
		float* input_args;
		int* accept, *best;
		double *density_output;
		thrust::complex<float>* input_greens;
		function* density;
		curandState* curand_states;
		elementaryChange* elem_changes;
	};

	//Temporary host buffers to store the pointers needed 
	//for freeing memory
	struct hostPointers{
		configuration* cur_conf;
		configuration* min_conf;
		configuration* tmp_conf;
		complexFunction* cur_greens;
		complexFunction* tmp_greens;
		complexFunction* eps_greens;
		complexFunction* epsh_greens;
		complexFunction* opt_greens;
		function* density;
	};

	/* 
	   Sets the minimum and the current configurations as the last step for a local update
	   Larger deviations might be accepted, (to be able to escape local minima) 
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 
	   */
	__device__ void setCurrentConfig(configuration* min, configuration* current,
			float* dev_current, float* dev_opt,
			float* dev_min, elementaryChange* change,
			float* eps, curandState* states, int* accept,
			int numLocal)
	{
		int idx = blockIdx.x;

		//If the optimal deviation is smaller than the previous
		if(dev_opt[idx] < dev_current[idx] && eps[idx] != D_EPS_ERROR)
		{
			accept[idx] = 1;
			updateConfig(current, change, eps);
		}
		else
		{
			float x = dev_current[idx] / dev_opt[idx];
			float d = numLocal < D_LOCAL_UPDATES/4 ? 1 : 20;
			float f = powf(x, d);
			float r = curand_uniform(&states[idx]);
			if(r < f && eps[idx] != D_EPS_ERROR)
			{
				accept[idx] = 1;
				updateConfig(current, change, eps);
			}
			else
			{
				accept[idx] = 0;
			}
		}
	}

	/* Sets the new minimum configuration if the deviation is smaller
	   The output configurations are stored in minimum, and the proposed configuration
	   in current and the corresponding deviations in dev, and dev_opt
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (1-512) x 1 x 1 */
	__device__ void setMinimumConfig(configuration* minimum, configuration* current, 
			float* dev_min, float* dev_opt) 
	{
		if(dev_opt[blockIdx.x] < dev_min[blockIdx.x])
		{
			if(threadIdx.x == 0) 
			{
				dev_min[blockIdx.x] = dev_opt[blockIdx.x];
				minimum[blockIdx.x].numRectangles = current[blockIdx.x].numRectangles;
			}
			__syncthreads();

			int numRectangles = current[blockIdx.x].numRectangles;
			int m = blockDim.x >= numRectangles ? numRectangles : blockDim.x;
			int i = threadIdx.x;

			while(i < numRectangles)
			{
				minimum[blockIdx.x].rectangles[i].w = current[blockIdx.x].rectangles[i].w; 
				minimum[blockIdx.x].rectangles[i].h = current[blockIdx.x].rectangles[i].h; 
				minimum[blockIdx.x].rectangles[i].c = current[blockIdx.x].rectangles[i].c;
				i += m;
			}
		}
	}

	/* Calculates the deviations given a known greens function and a approximated one
	 * Calculates the deviations (stored in dev) between the known greens function "input_greens" and
	 * approximated greens functions "approxGC"
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (Number of greens points) x 1 x 1 */
	__global__ void getDeviation(float* dev, thrust::complex<float>* input_greens, complexFunction* approxGC)
	{
		if(threadIdx.x == 0)
		{
			dev[blockIdx.x] = 0;
		}
		__syncthreads();

		thrust::complex<float> gf = input_greens[threadIdx.x];
		thrust::complex<float> agf = approxGC[blockIdx.x].functionValues[threadIdx.x];
		float deltaM = thrust::abs((gf - agf));

		atomicAdd(&dev[blockIdx.x], deltaM);
	}


	/*
	   Device function of getDeviation (identical)
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (Number of greens points) x 1 x 1 
	   */
	__device__ void getDeviation2(float* dev, thrust::complex<float>* input_greens, complexFunction* approxGC)
	{
		if(threadIdx.x == 0)
		{
			dev[blockIdx.x] = 0;
		}
		__syncthreads();

		thrust::complex<float> gf = input_greens[threadIdx.x];
		thrust::complex<float> agf = approxGC[blockIdx.x].functionValues[threadIdx.x];
		float deltaM = thrust::abs((gf - agf));
		atomicAdd(&dev[blockIdx.x], deltaM);
	}

	/* Specific analytic integral for kernel 0*/
	__device__ thrust::complex<float> analyticIntegral(float omega,  float c,  float w, float h)
	{
		thrust::complex<float> i(0.0f, 1.0);
		float a = c - w / 2;
		float b = c + w / 2;
		thrust::complex<float> K = (i * omega - a) / (i * omega - b);
		float x = K.real();
		float y = K.imag();
		return h*(__logf(sqrtf(x*x + y*y)) + i * atan2f(y,x)); 
	}


	/* Calculates the numerical integral using booles rule*/
	__device__ thrust::complex<float> numericalIntegral(float omega,  float c,  float w, float h)
	{
		kernelFunction k = kernelList[D_KERNEL];
		//Boole's rule
		float x1 = c - w / 2;
		float dh = w / 4;
		return h * 0.044444f * dh * 
			(7.0f * k(omega, x1) + 32.0f * k(omega, x1 + dh) + 12.0f * k(omega, x1 + 2 * dh) +
			 32.0f * k(omega, x1 + 3 * dh)+ 7.0f * k(omega, x1 + 4 * dh));
	}


	/* 
	 * Compute the integral, analytical if using kernel0 
	 * */
	typedef thrust::complex<float> (*integralFunction)(float,float,float,float);
	//List of the integral functions, analytical and numerical.
	__constant__ integralFunction ilist[2] = {analyticIntegral, numericalIntegral};

	__device__ thrust::complex<float> getIntegral(float omega,  float c,  float w, float h)
	{
		return ilist[(D_KERNEL == 0) ? 0 : 1](omega, c, w, h);
	}

	/* Calculates the greens function with added change given a configuration 
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (Number of greens points) x 1 x 1 */
	__global__ void getGreens(complexFunction* approxGC, configuration* config, float* omegas) 
	{
		float omega = omegas[threadIdx.x];
		thrust::complex<float> sum = 0;
		float w, h, c;
		for(int i = 0; i < config[blockIdx.x].numRectangles; i++)
		{
			w = config[blockIdx.x].rectangles[i].w;
			h = config[blockIdx.x].rectangles[i].h;
			c = config[blockIdx.x].rectangles[i].c;

			//Calculate the integral (numerical)
			sum += getIntegral(omega, c, w, h);
		}
		approxGC[blockIdx.x].functionValues[threadIdx.x] = sum;
	}

	/* Calculates the greens function with an added change, given a configuration.
	   (For all global configurations in parallel)
	   It does not traverse over all rectangles, it is based on an old greens function
	   and changes it depending on the proposed change. 
	   the output greens function is stored in "new_greens"
	   "old_greens" holds the previous Greens function
	   "config" stores the configurations where the elementary changes are done upon
	   "eps" holds the sizes of the changes
	   "greens_args" holds the arguments where the Greens function is defined
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (Number of greens points) x 1 x 1 */
	__device__ void getGreens2(complexFunction* old_greens, configuration* config,
			elementaryChange* changes, float* eps, 
			complexFunction* new_greens, float* greens_args)
	{
		float omega = greens_args[threadIdx.x];
		int i, j;
		thrust::complex<float> change = 0;
		float w, h, c;
		int type = changes[blockIdx.x].type;

		i = changes[blockIdx.x].rect1;
		w = config[blockIdx.x].rectangles[i].w;
		h = config[blockIdx.x].rectangles[i].h;
		c = config[blockIdx.x].rectangles[i].c;

		switch (type) 
		{
			case SHIFT:
				{
					change -= getIntegral(omega, c, w, h);
					change += getIntegral(omega, c + eps[blockIdx.x], w, h);   
					break;
				}
			case WIDTH:
				{
					change -= getIntegral(omega, c, w, h);
					float weight = w * h;
					w += eps[blockIdx.x];
					h = weight/w;
					change += getIntegral(omega, c, w, h);
					break;
				}
			case WEIGHT:
				{
					j = changes[blockIdx.x].rect2;
					change -= getIntegral(omega, c, w, h);
					h += eps[blockIdx.x];
					change += getIntegral(omega, c, w, h);

					w = config[blockIdx.x].rectangles[j].w;
					h = config[blockIdx.x].rectangles[j].h;
					c = config[blockIdx.x].rectangles[j].c;
					change -= getIntegral(omega, c, w, h);

					float wo = config[blockIdx.x].rectangles[i].w;
					float wt = config[blockIdx.x].rectangles[j].w;
					h -= eps[blockIdx.x]*wo/wt;
					change += getIntegral(omega, c, w, h);
					break;
				}
			case ADD:
				{
					float oldWeight = config[blockIdx.x].rectangles[i].w*config[blockIdx.x].rectangles[i].h;
					float r = (eps[blockIdx.x] - D_WEIGHT_MIN)/(oldWeight - 2*D_WEIGHT_MIN);
					float cnew = (D_OMEGA_MIN + D_WEIGHT_MIN/2) + (D_OMEGA_MAX - D_OMEGA_MIN - 
							D_WEIGHT_MIN)*changes[blockIdx.x].r;
					float wmax = 2*min(D_OMEGA_MAX - cnew, cnew - D_OMEGA_MIN);
					float hnew = eps[blockIdx.x]/wmax + r*(eps[blockIdx.x]/D_WEIGHT_MIN - eps[blockIdx.x]/wmax);
					float wnew = eps[blockIdx.x]/hnew;

					//If adding a rectangle is possible
					if (!(oldWeight <= 2*D_WEIGHT_MIN || eps[blockIdx.x] < D_WEIGHT_MIN || 
								eps[blockIdx.x] > (oldWeight-D_WEIGHT_MIN))) 
					{
						change += getIntegral(omega, cnew, wnew, hnew);
						change -= getIntegral(omega, c, w, h);
						float w2 = config[blockIdx.x].rectangles[i].w;
						h = (oldWeight - eps[blockIdx.x]) / w2;
						change += getIntegral(omega, c, w, h);
					}
					break;
				}
			case REMOVE:
				{
					change -= getIntegral(omega, c, w, h);
					j = changes[blockIdx.x].rect2;
					w = config[blockIdx.x].rectangles[j].w;
					h = config[blockIdx.x].rectangles[j].h;
					c = config[blockIdx.x].rectangles[j].c;
					change -= getIntegral(omega, c, w, h);
					float weight = config[blockIdx.x].rectangles[i].w*config[blockIdx.x].rectangles[i].h;
					h += weight/config[blockIdx.x].rectangles[j].w;
					c += eps[blockIdx.x];
					change += getIntegral(omega, c, w, h);
					break;
				}
			case SPLIT:
				{
					float w1 = D_WEIGHT_MIN + changes[blockIdx.x].r*(w - D_WEIGHT_MIN);
					float w2 = w - w1;
					//If split actually is possible
					if(!(w <= 2*D_WEIGHT_MIN || w1*h < D_WEIGHT_MIN || w2*h < D_WEIGHT_MIN || w1 < 0 || w2 < 0)){
						change += getIntegral(omega, c - eps[blockIdx.x], w2, h);
						change -= getIntegral(omega, c, w, h);
						w = w1;
						c = c + eps[blockIdx.x];
						change += getIntegral(omega, c, w, h);
					}
					break;
				}
			case GLUE:
				{
					change -= getIntegral(omega, c, w, h);
					float wt = config[blockIdx.x].rectangles[changes[blockIdx.x].rect2].w;
					float ht = config[blockIdx.x].rectangles[changes[blockIdx.x].rect2].h;
					float wnew = (w + wt)/2;
					float weight = w*h + wt*ht;
					float hnew = weight/wnew;
					w = wnew;
					h = hnew;
					c = (c + config[blockIdx.x].rectangles[changes[blockIdx.x].rect2].c)/2;
					c += eps[blockIdx.x];
					change += getIntegral(omega, c, w, h);

					//Remove other rectangle
					j = changes[blockIdx.x].rect2;
					w = config[blockIdx.x].rectangles[j].w;
					h = config[blockIdx.x].rectangles[j].h;
					c = config[blockIdx.x].rectangles[j].c;
					change -= getIntegral(omega, c, w, h);
					break;
				}
		}
		thrust::complex<float> next = old_greens[blockIdx.x].functionValues[threadIdx.x] + change;

		new_greens[blockIdx.x].functionValues[threadIdx.x] = next;
	}


	/*  Calculates the optimal sizes of the changes (using an interpolated parabola) and stores 
	 *  them in eps_opt
		Grid size= (Number of global updates) x 1 x 1
		Block size= 1 x 1 x 1 */
	__device__ void getOptimalEpsilon(float* eps, float* eps_half, float* eps_opt,
			float* dev_current, float* dev_eps_half,
			float* dev_eps, elementaryChange* change)
	{
		int idx = blockIdx.x;

		float A = 2*(dev_eps[idx] - 2*dev_eps_half[idx] + dev_current[idx])/(eps[idx]*eps[idx]);
		float B = (4*dev_eps_half[idx] - dev_eps[idx] - 3*dev_current[idx])*eps[idx];

		if(A <= 0)
		{
			eps_opt[idx] = D_EPS_ERROR;
		}
		else
		{
			eps_opt[idx] = -B/(2*A);
			if(eps_opt[idx] < change[idx].emin || eps_opt[idx] > change[idx].emax)
			{
				eps_opt[idx] = D_EPS_ERROR;
			}
		}
	}

	/* Checks which eps actually gives the minimal deviation out of eps, eps_half and eps_opt 
	   The resulting eps are stored in eps_opt
	   best holds integers which represents which of the three eps was the best.
	   best = 0 corresponds to eps, best = 1 to eps_half and best = 2 to eps_opt
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 */
	__device__ void setOptimalConfiguration(float* dev_eps, float* dev_eps_half, 
			float* dev_eps_opt, float* eps,
			float* eps_half, float* eps_opt, 
			elementaryChange* change, int* best) 
	{
		int idx = blockIdx.x;

		//if proposed optimal is inside emin, emax and is actually the smallest and no error
		if(dev_eps_opt[idx] < dev_eps[idx] && dev_eps_opt[idx] < dev_eps_half[idx] && 
				eps_opt[idx] != D_EPS_ERROR)
		{
			best[idx] = 2;
		}
		else if(dev_eps[idx] < dev_eps_half[idx] || (eps_half[idx] < change[idx].emin ||
					eps_half[idx] > change[idx].emax) )
		{
			best[idx] = 0;
			eps_opt[idx] = eps[idx];
			dev_eps_opt[idx] = dev_eps[idx];
		}
		else
		{
			best[idx] = 1;
			eps_opt[idx] = eps_half[idx];
			dev_eps_opt[idx] = dev_eps_half[idx];
		}
	}

	/* Copies the minimum configuration to the current configuration
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 */
	__global__ void  copyMinimumToCurrentConfig(configuration* min, configuration* cur)
	{
		copyConfiguration(cur, min);
	}

	/* Computes the density function given a configuration 
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= (1-512) x 1 x 1 */
	__global__ void computeDensityFunction(configuration* configs, function* density)
	{
		//	int m = D_DENSITY_FUNCTION_POINTS/D_BLOCK_SIZE;
		double c, w, h, sum, omega;

		int numPoints = D_DENSITY_FUNCTION_POINTS;
		int m = blockDim.x >= numPoints ? numPoints : blockDim.x;
		int i = threadIdx.x;

		while (i < numPoints) 
		{
			sum = 0;
			for(int j = 0; j < configs[blockIdx.x].numRectangles; j++)
			{
				omega = D_OMEGA_MIN + (double)i/(D_DENSITY_FUNCTION_POINTS-1)*(D_OMEGA_MAX - D_OMEGA_MIN);
				c = configs[blockIdx.x].rectangles[j].c;
				w = configs[blockIdx.x].rectangles[j].w;
				h = configs[blockIdx.x].rectangles[j].h;
				if(omega <= (c + w/2) && omega >= (c - w/2))
				{
					sum += h;
				}
			}
			density[blockIdx.x].functionValues[i] = sum ;
			i += m;
		}
	}

	/* Computes the average density function over all global updates
	   Grid size= 1 x 1 x 1
	   Block size= 1 x 1 x 1 */
	__global__ void averageDensityFunction(double* result, function* density)
	{
		double sum = 0;

		for(int i = 0; i < D_DENSITY_FUNCTION_POINTS; i++)
		{
			sum = 0;
			for(int j = 0; j < D_GRID_SIZE; j++)
			{
				sum += density[j].functionValues[i];
			}
			result[i] = sum/D_GRID_SIZE;
		}
	}

	/* Copies the Greens function in tempGC to GC, if accept = 1
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 */
	__device__ void setCurrentGreens(complexFunction* GC, complexFunction* tempGCs, int* accept) 
	{
		if(accept[blockIdx.x] == 1) 
		{
			GC[blockIdx.x].functionValues[threadIdx.x] = tempGCs[blockIdx.x].functionValues[threadIdx.x];
		}
	}


	/*
	 * Copies the output from GPU memory to "out", and calculates the 
	 * corresponding output arguments to "out_args"
	 * */
	void copyFinalFunction(devicePointers &DP, float* out, float* out_args)
	{
		double *P = (double*)malloc(sizeof(double)*H_DENSITY_FUNCTION_POINTS);
		gpuErrchk(cudaMemcpy(P, DP.density_output, sizeof(double)*H_DENSITY_FUNCTION_POINTS,
					cudaMemcpyDeviceToHost));
		for(int i = 0; i < H_DENSITY_FUNCTION_POINTS; i++)
		{
			out[i] = P[i];
			out_args[i] = H_OMEGA_MIN + 
				(float)i/(H_DENSITY_FUNCTION_POINTS - 1)*(H_OMEGA_MAX - H_OMEGA_MIN);
		}
		free(P);
	}

	/*
	 * Puts the actual best greens function in tempGC
	 * Grid size= (Number of global updates x 1 x 1)
	 * Block size=(Number of input greens points x 1 x 1)
	 * */
	__device__ void setTempGC(complexFunction* tempGC, complexFunction *eps_greens,
			complexFunction* epsh_greens, complexFunction* opt_greens, int *best)
	{
		if(best[blockIdx.x] == 0){
			tempGC[blockIdx.x].functionValues[threadIdx.x] = eps_greens[blockIdx.x].functionValues[threadIdx.x];
		}
		else if(best[blockIdx.x] == 1)
		{
			tempGC[blockIdx.x].functionValues[threadIdx.x] = epsh_greens[blockIdx.x].functionValues[threadIdx.x];
		}
		else
		{
			tempGC[blockIdx.x].functionValues[threadIdx.x] = opt_greens[blockIdx.x].functionValues[threadIdx.x];
		}
	}

	/*
	 * Computes N elementary updates for all global configurations stored in DP.cur_conf
	 * "DP" is a struct which holds all needed device (gpu) memory buffers
	 * Most device buffers have the same length as the number of global configurations.
	 * j is the current local update
	 * The minimum of all elementary updates are stored in DP.min_conf 
	 * Grid size= (Number of global updates x 1 x 1)
	 * Block size= (Number of input green's points x 1 x 1)
	 */
	__global__ void step(devicePointers DP, int j, int N)
	{ 
		for(int i = 0; i < N; i++){

			//Generate a new random elementary change for all global configurations
			//The size of the change is stored in DP.eps and DP.epsh is half of that.
			if(threadIdx.x == 0){
				generateRandomChange(DP.elem_changes, DP.curand_states, DP.cur_conf, DP.eps, DP.epsh);
			}
			__syncthreads();

			//Compute the Green functions with the added change for both eps and half eps. With these an optimal eps
			//is calculated in getOptimalEpsilon.
			//In PRB 62 6317 (2000) appendix B3, eps is called delta ksi.
			getGreens2(DP.cur_greens, DP.cur_conf, DP.elem_changes, DP.eps, DP.eps_greens, DP.input_args);
			__syncthreads();
			getGreens2(DP.cur_greens, DP.cur_conf, DP.elem_changes, DP.epsh, DP.epsh_greens, DP.input_args);
			__syncthreads();

			//Compute the error/deviation from the current configuration and with the changes
			getDeviation2(DP.cur_dev, DP.input_greens, DP.cur_greens);
			getDeviation2(DP.eps_dev, DP.input_greens, DP.eps_greens);
			getDeviation2(DP.epsh_dev, DP.input_greens, DP.epsh_greens);
			__syncthreads();

			//Compute the "optimal" change size, assuming the local minima is quadratic
			//As suggested in PRB 62 6317 (2000) equation B12-B14
			if(threadIdx.x == 0)
				getOptimalEpsilon(DP.eps, DP.epsh, DP.opt_eps, DP.cur_dev, DP.epsh_dev, DP.eps_dev, DP.elem_changes);
			__syncthreads();

			//Compute the greens function for the "optimal" change size
			getGreens2(DP.cur_greens, DP.cur_conf, DP.elem_changes, DP.opt_eps, DP.opt_greens, DP.input_args);
			__syncthreads();
			getDeviation2(DP.opt_dev, DP.input_greens, DP.opt_greens);
			__syncthreads();

			//See which of the proposed changes between eps, half eps and "optimal eps" gives the smallest error/deviation
			if(threadIdx.x == 0)
				setOptimalConfiguration(DP.eps_dev, DP.epsh_dev, DP.opt_dev, DP.eps, DP.epsh, DP.opt_eps, DP.elem_changes, 
						DP.best);
			__syncthreads();

			//Move the best Greens function to tmp_greens
			setTempGC(DP.tmp_greens, DP.eps_greens, DP.epsh_greens, DP.opt_greens, DP.best);
			__syncthreads();

			//Set the new configuration
			if(threadIdx.x == 0)
				setCurrentConfig(DP.min_conf, DP.cur_conf, DP.cur_dev, DP.opt_dev, DP.min_dev, DP.elem_changes, DP.opt_eps,
						DP.curand_states, DP.accept, j);
			__syncthreads();

			//Set the new current greens function
			setCurrentGreens(DP.cur_greens, DP.tmp_greens, DP.accept);
			__syncthreads();

			//Check if the new configuration is the minimal in this local update
			//And move it to DP.min_conf
			setMinimumConfig(DP.min_conf, DP.cur_conf, DP.min_dev, DP.cur_dev);
			__syncthreads();
		}
	}

	/*
	 * Create the random initial configurations
	 * */
	void initConfigurations(devicePointers DP, hostPointers HP)
	{
		//Random initial configuration
		createRandomConfiguration<<<H_GRID_SIZE, 1>>>(DP.cur_conf, 
				DP.curand_states, H_STARTING_RECTANGLES);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// Set current and minimum configurations as the initial
		copy<<<H_GRID_SIZE, 1>>>(DP.min_conf, DP.cur_conf);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		getGreens<<<H_GRID_SIZE, H_BLOCK_SIZE>>>(DP.cur_greens, DP.min_conf, DP.input_args);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		getDeviation<<<H_GRID_SIZE, H_GREENS_FUNCTION_POINTS>>>(DP.min_dev, DP.input_greens, DP.cur_greens); //Min
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}

	/* Initiates Curand for random numbers */
	__global__ void initCurand(curandState* states)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		curand_init(111, idx, 0, &states[idx]); //111 = seed
	}

	/* Calculates the average error/deviation over all
	 * global configurations 
	 * Grid size= (1 x 1 x 1)
	 * Block size= (1 x 1 x 1)
	 * */
	__global__ void getAvgDeviation(devicePointers DP, float* avg_res)
	{
		float avg = 0;
		for(int i = 0; i < D_GRID_SIZE; i++)
		{
			avg += DP.min_dev[i];	
		}
		*avg_res = avg/D_GRID_SIZE;
	}

	float averageDeviation(devicePointers DP) {
		float* avgDev; 
		float res[1];
		gpuErrchk(cudaMalloc((void**)&avgDev, sizeof(float)));
		getAvgDeviation<<<1, 1>>>(DP, avgDev);
		cudaError_t err = cudaMemcpy(res, avgDev, sizeof(float),cudaMemcpyDeviceToHost);
		gpuErrchk(cudaFree(avgDev));
		return *res;
	}

	void assignConstants(input &inp)
	{
		H_STARTING_RECTANGLES = inp.starting_rectangles;
		H_DENSITY_FUNCTION_POINTS = inp.num_output_points;
		H_GREENS_FUNCTION_POINTS = inp.num_input_points;
		H_GRID_SIZE = inp.num_global;
		H_BLOCK_SIZE = H_GREENS_FUNCTION_POINTS;	
		H_LOCAL_UPDATES = inp.num_local;
		H_ELEMENTARY_UPDATES = inp.num_elementary;
		H_KERNEL = inp.kernel;
		H_OMEGA_MAX = inp.max_omega;
		H_OMEGA_MIN = inp.min_omega;
		H_MAX_RECTANGLES = H_STARTING_RECTANGLES*2;

		gpuErrchk(cudaMemcpyToSymbol(D_MAX_RECTANGLES, &H_MAX_RECTANGLES, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_DENSITY_FUNCTION_POINTS, &H_DENSITY_FUNCTION_POINTS, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_GREENS_FUNCTION_POINTS, &H_GREENS_FUNCTION_POINTS, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_GRID_SIZE, &H_GRID_SIZE, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_BLOCK_SIZE, &H_BLOCK_SIZE, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_LOCAL_UPDATES, &H_LOCAL_UPDATES, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_ELEMENTARY_UPDATES, &H_ELEMENTARY_UPDATES, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_KERNEL, &H_KERNEL, sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(D_BETA, &inp.beta, sizeof(float)));
		gpuErrchk(cudaMemcpyToSymbol(D_OMEGA_MAX, &H_OMEGA_MAX, sizeof(float)));
		gpuErrchk(cudaMemcpyToSymbol(D_OMEGA_MIN, &H_OMEGA_MIN, sizeof(float)));

	}

	/*
	   Allocates the buffers defined in the structs DP and HP
	   */
	void allocateBuffers(devicePointers &DP, hostPointers &HP, input &inp)
	{
		const int fsize = sizeof(function) * H_GRID_SIZE;
		const int csize = sizeof(configuration) * H_GRID_SIZE;
		const int fcsize = sizeof(complexFunction) * H_GRID_SIZE;

		//Allocate configurations
		HP.cur_conf = (configuration*)malloc(csize);
		gpuErrchk(cudaMalloc((void**)&DP.cur_conf, csize));
		HP.min_conf = (configuration*)malloc(csize);
		gpuErrchk(cudaMalloc((void**)&DP.min_conf, csize));
		HP.tmp_conf = (configuration*)malloc(csize);

		//Allocate functions
		HP.cur_greens = (complexFunction*)malloc(fcsize);
		gpuErrchk(cudaMalloc((void**)&DP.cur_greens, fcsize));
		HP.density = (function*)malloc(fsize);
		gpuErrchk(cudaMalloc((void**)&DP.density, fsize));
		HP.tmp_greens = (complexFunction*)malloc(fcsize); 
		gpuErrchk(cudaMalloc((void**)&DP.tmp_greens, fcsize));
		HP.eps_greens = (complexFunction*)malloc(fcsize); 
		gpuErrchk(cudaMalloc((void**)&DP.eps_greens, fcsize));

		HP.epsh_greens = (complexFunction*)malloc(fcsize); 
		gpuErrchk(cudaMalloc((void**)&DP.epsh_greens, fcsize));

		HP.opt_greens = (complexFunction*)malloc(fcsize); 
		gpuErrchk(cudaMalloc((void**)&DP.opt_greens, fcsize));


		for(int i = 0; i < H_GRID_SIZE; i++)
		{
			HP.cur_conf[i].numRectangles = 0;
			HP.min_conf[i].numRectangles = 0;
			HP.tmp_conf[i].numRectangles = 0;

			gpuErrchk(cudaMalloc((void**)&HP.cur_conf[i].rectangles, sizeof(rectangle)*H_MAX_RECTANGLES));
			gpuErrchk(cudaMalloc((void**)&HP.min_conf[i].rectangles, sizeof(rectangle)*H_MAX_RECTANGLES));
			gpuErrchk(cudaMalloc((void**)&HP.tmp_conf[i].rectangles, sizeof(rectangle)*H_MAX_RECTANGLES));

			gpuErrchk(cudaMalloc((void**)&HP.cur_greens[i].functionValues, sizeof(thrust::complex<float>) *H_GREENS_FUNCTION_POINTS));
			gpuErrchk(cudaMalloc((void**)&HP.density[i].functionValues, sizeof(float) * H_DENSITY_FUNCTION_POINTS));
			gpuErrchk(cudaMalloc((void**)&HP.tmp_greens[i].functionValues, sizeof(thrust::complex<float>) * H_GREENS_FUNCTION_POINTS));
			gpuErrchk(cudaMalloc((void**)&HP.eps_greens[i].functionValues, sizeof(thrust::complex<float>) * H_GREENS_FUNCTION_POINTS));
			gpuErrchk(cudaMalloc((void**)&HP.epsh_greens[i].functionValues, sizeof(thrust::complex<float>) * H_GREENS_FUNCTION_POINTS));
			gpuErrchk(cudaMalloc((void**)&HP.opt_greens[i].functionValues, sizeof(thrust::complex<float>) * H_GREENS_FUNCTION_POINTS));
		}

		gpuErrchk(cudaMemcpy(DP.cur_conf, HP.cur_conf, csize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.min_conf, HP.min_conf, csize, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(DP.cur_greens, HP.cur_greens, fcsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.tmp_greens, HP.tmp_greens, fcsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.eps_greens, HP.eps_greens, fcsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.epsh_greens, HP.epsh_greens, fcsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.opt_greens, HP.opt_greens, fcsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.density, HP.density, fsize, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc((void**)&DP.input_greens, H_GREENS_FUNCTION_POINTS * sizeof(thrust::complex<float>)));
		gpuErrchk(cudaMalloc((void**)&DP.input_args, H_GREENS_FUNCTION_POINTS * sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.density_output, H_DENSITY_FUNCTION_POINTS*sizeof(double)));

		gpuErrchk(cudaMalloc((void**)&DP.cur_dev, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.epsh_dev, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.eps_dev, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.min_dev, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.opt_dev, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.eps, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.epsh, H_GRID_SIZE*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&DP.opt_eps, H_GRID_SIZE*sizeof(float)));

		gpuErrchk(cudaMalloc((void**)&DP.elem_changes, sizeof(elementaryChange)*H_GRID_SIZE));
		gpuErrchk(cudaMalloc((void**)&DP.accept, sizeof(int)*H_GRID_SIZE));
		gpuErrchk(cudaMalloc((void**)&DP.best, sizeof(int)*H_GRID_SIZE));
		gpuErrchk(cudaMalloc(&DP.curand_states, sizeof(curandState) * H_GRID_SIZE));
	}


	/*
	 * Copies the greens function defined in inp to device (gpu) memory
	 * in DP.input_greens
	 * */
	void assignBuffers(devicePointers &DP, hostPointers &HP, input &inp)
	{
		int ssize = sizeof(thrust::complex<float>);
		thrust::complex<float> *G = (thrust::complex<float>*)malloc(ssize*inp.num_input_points);
		for(int i = 0; i < inp.num_input_points; i++)
		{
			G[i].real(inp.input_greens_real[i]);	
			G[i].imag(inp.input_greens_imag[i]);	
		}
		gpuErrchk(cudaMemcpy(DP.input_greens, G, ssize * inp.num_input_points, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(DP.input_args, inp.input_args, sizeof(float) * inp.num_input_points, cudaMemcpyHostToDevice));
		free(G);

	}

	/*
	   Frees all buffers
	   */
	void freeBuffers(devicePointers DP, hostPointers HP)
	{
		for(int i = 0; i < H_GRID_SIZE; i++)
		{
			gpuErrchk(cudaFree(HP.cur_conf[i].rectangles));
			gpuErrchk(cudaFree(HP.min_conf[i].rectangles));
			gpuErrchk(cudaFree(HP.tmp_conf[i].rectangles));

			gpuErrchk(cudaFree(HP.cur_greens[i].functionValues));
			gpuErrchk(cudaFree(HP.tmp_greens[i].functionValues));
			gpuErrchk(cudaFree(HP.eps_greens[i].functionValues));
			gpuErrchk(cudaFree(HP.epsh_greens[i].functionValues));
			gpuErrchk(cudaFree(HP.opt_greens[i].functionValues));
			gpuErrchk(cudaFree(HP.density[i].functionValues));
		}

		gpuErrchk(cudaFree(DP.cur_conf));
		gpuErrchk(cudaFree(DP.min_conf));
		gpuErrchk(cudaFree(DP.density));
		gpuErrchk(cudaFree(DP.cur_greens));
		gpuErrchk(cudaFree(DP.tmp_greens));
		gpuErrchk(cudaFree(DP.eps_greens));
		gpuErrchk(cudaFree(DP.epsh_greens));
		gpuErrchk(cudaFree(DP.opt_greens));

		free(HP.cur_conf);
		free(HP.min_conf);
		free(HP.tmp_conf);
		free(HP.cur_greens);
		free(HP.eps_greens);
		free(HP.epsh_greens);
		free(HP.opt_greens);
		free(HP.density);

		gpuErrchk(cudaFree(DP.density_output));
		gpuErrchk(cudaFree(DP.input_greens));
		gpuErrchk(cudaFree(DP.input_args));
		gpuErrchk(cudaFree(DP.cur_dev));
		gpuErrchk(cudaFree(DP.epsh_dev));
		gpuErrchk(cudaFree(DP.eps_dev));
		gpuErrchk(cudaFree(DP.min_dev));
		gpuErrchk(cudaFree(DP.opt_dev));
		gpuErrchk(cudaFree(DP.eps));
		gpuErrchk(cudaFree(DP.epsh));
		gpuErrchk(cudaFree(DP.opt_eps));
		gpuErrchk(cudaFree(DP.elem_changes));
		gpuErrchk(cudaFree(DP.accept));
		gpuErrchk(cudaFree(DP.best));
		gpuErrchk(cudaFree(DP.curand_states));
	}

	/* 
	   Prints the GPL license message 
	   */
	void printGPL() 
	{
		printf("GPU Stochastic Optimization Method\n"
				"Copyright (C) 2016 Johan Nordström\n"
				"This program comes with ABSOLUTELY NO WARRANTY;\n"
				"This is free software, and you are welcome to redistribute it\n"
				"under the terms of the GNU General Public License;\n");
	}


	/*
	 * Computes spectral function given a Green's function and a 
	 * kernel. The algorithm is described by Mischenko et.al in PRB 62 6317 (2000) appendix B3.
	 * The input data and parameters are stored in "inp".  
	 * The output spectral function is stored in "output".
	 * The output arguments, (I.e where the spectral function is defined is stored in "output_args".
	 */
	void compute(input &inp, float* output, float* output_args)
	{
		//Structure that holds all pointers to device memory
		//The used device pointers
		devicePointers DP;
		hostPointers HP;

		//Read the input and assign to device buffers
		assignConstants(inp);
		allocateBuffers(DP, HP, inp);
		assignBuffers(DP, HP, inp);

		//Init curand for random numbers
		initCurand<<<H_GRID_SIZE, 1>>>(DP.curand_states);

		// Create initial configurations
		initConfigurations(DP, HP);

		//Print the GPL license
		printGPL();

		int N = min(64, H_ELEMENTARY_UPDATES); //64 is an empirical chosen batch size 
		int N2 = (H_ELEMENTARY_UPDATES % N == 0) ? N : H_ELEMENTARY_UPDATES % N; //Remainder of the batch 
		for(int j = 0; j < H_LOCAL_UPDATES; j++)
		{
			//Compute the current Green's function
			getGreens<<<H_GRID_SIZE, H_BLOCK_SIZE>>>(DP.cur_greens, DP.cur_conf, DP.input_args);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			//Compute N elementary updates in a batch to reduce call overhead
			for(int i = 0; (i+N) < H_ELEMENTARY_UPDATES; i+=N)
			{
				step<<<H_GRID_SIZE, H_BLOCK_SIZE>>>(DP, j, N);
				gpuErrchk( cudaPeekAtLastError() );
				gpuErrchk( cudaDeviceSynchronize() );
			}
			step<<<H_GRID_SIZE, H_BLOCK_SIZE>>>(DP, j, N2); //Remainder of the batch
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			//Copy the minimum configuration as the start for the next local update
			copyMinimumToCurrentConfig<<<H_GRID_SIZE, 1>>>(DP.min_conf, DP.cur_conf);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			//Calculate the average error/deviation and print it
			if(j % 20 == 0)
			{
				float avg = averageDeviation(DP);
				printf("Local run %d/%d. Avg. deviation: %f \n", j + 1, H_LOCAL_UPDATES, avg);
			}
		}
		
		//Print the last average error/deviation
		float avg = averageDeviation(DP);
		printf("Local run %d/%d. Avg. deviation: %f \n", H_LOCAL_UPDATES, H_LOCAL_UPDATES, avg);
		
		//Compute the spectral functions (density functions) for all global updates 
		computeDensityFunction<<<H_GRID_SIZE, H_BLOCK_SIZE>>>(DP.min_conf, DP.density);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		//Average the functions for all global configurations for the final answer 
		averageDensityFunction<<<1,1>>>(DP.density_output, DP.density);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		//Copy the final function to output
		copyFinalFunction(DP, output, output_args);

		//Free all memory
		freeBuffers(DP, HP);

		cudaDeviceReset();
	}

}//end namespace
