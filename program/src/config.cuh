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

#ifndef __CONFIG_CUH__
#define __CONFIG_CUH__

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/complex.h>

namespace mish
{

	/* Elementary update enum */
	enum {
		SHIFT, WIDTH, WEIGHT, ADD, REMOVE, SPLIT, GLUE
	};

	/* Rectangle struct */
	struct rectangle
	{
		float w, h, c;
	};

	/* Configuration struct */
	struct configuration
	{
		int numRectangles;
		rectangle* rectangles;
	};

	/* Function struct */
	struct function
	{
		float* functionValues;
	};

	/* Complex function struct */
	struct complexFunction
	{
		thrust::complex<float> *functionValues;
	};

	/* Helper struct for storing information about an elementary update */
	struct elementaryChange
	{
		int type; //One of the enum changes
		int rect1; //First affected rectangle (per index)
		int rect2; //Second affected
		float emin; //Minimum allowed epsilon
		float emax; //Maximum allowed epsilon
		float r; //Random number needed for splitting
	};

	void printConfigs(configuration* configs, configuration* h_temp);

	__global__ void copy(configuration* tar, configuration* s);
	__global__ void checkConstraints(configuration* config, elementaryChange* change);
	__global__ void totalWeight(configuration* config, double* weight);
	__global__ void createRandomConfiguration(configuration* currentConfig, curandState* states, int numRectangles);

	__device__ void generateRandomChange(elementaryChange* changes, curandState* states,
			configuration* configs, float* eps1, float* eps2);

	__device__ void copyConfiguration(configuration* target, configuration* src);
	__device__ void addShift(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addRectangle(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addWidth(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addWeight(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addRemove(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addRemove(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addSplit(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void addGlue(elementaryChange* changes, configuration* configs, float* eps);
	__device__ void updateConfig(configuration* current, elementaryChange* change, float* eps);
}
#endif
