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

#include "config.cuh"
#include "constants.cuh"
#include <stdio.h>
#include <cuda.h>

namespace mish
{

	/* Copies a configuration from one memory address to another
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 */
	__global__ void copy(configuration* tar, configuration* s)
	{
		copyConfiguration(tar, s);
	}

	/* Copies a configuration from one memory address to another
	   Grid size= (Number of global updates) x 1 x 1
	   Block size= 1 x 1 x 1 */
	__device__ void copyConfiguration(configuration* target, configuration* src)
	{
		target[blockIdx.x].numRectangles = src[blockIdx.x].numRectangles;
		int i;
		for(i = 0; i < src[blockIdx.x].numRectangles; i++)
		{
			target[blockIdx.x].rectangles[i].w = src[blockIdx.x].rectangles[i].w; 
			target[blockIdx.x].rectangles[i].h = src[blockIdx.x].rectangles[i].h; 
			target[blockIdx.x].rectangles[i].c = src[blockIdx.x].rectangles[i].c; 
		}
	}

	/* Shifts rectangle in a configuration */
	__device__ void addShift(elementaryChange* changes, configuration* configs, float* eps)
	{
		float c = configs[blockIdx.x].rectangles[changes[blockIdx.x].rect1].c;

		//Do not shift if its outside
		if((c + eps[blockIdx.x] < D_OMEGA_MAX) && (c + eps[blockIdx.x] > D_OMEGA_MIN))
		{
			configs[blockIdx.x].rectangles[changes[blockIdx.x].rect1].c += eps[blockIdx.x];
		}

	}

	/* Adds a rectangle to a configuration */
	__device__ void addRectangle(elementaryChange* changes, configuration* configs, float* eps)
	{
		int idx = blockIdx.x;

		float oldWeight = configs[idx].rectangles[changes[idx].rect1].w*configs[idx].rectangles[changes[idx].rect1].h;

		//If any of these holds, it is not possible to add a rectangle
		if (oldWeight <= 2*D_WEIGHT_MIN || eps[idx] < D_WEIGHT_MIN || eps[idx] > (oldWeight-D_WEIGHT_MIN)) 
		{
			return;
		}

		float r = (eps[idx] - D_WEIGHT_MIN)/(oldWeight - 2*D_WEIGHT_MIN);
		float cnew = (D_OMEGA_MIN + D_WIDTH_MIN/2) + (D_OMEGA_MAX - D_OMEGA_MIN - D_WIDTH_MIN)*changes[idx].r;
		float wmax = 2*min(D_OMEGA_MAX - cnew, cnew - D_OMEGA_MIN);
		float hnew = eps[idx]/wmax + r*(eps[idx]/D_WIDTH_MIN - eps[idx]/wmax);
		float wnew = eps[idx]/hnew;
		configs[idx].numRectangles++;
		configs[idx].rectangles[configs[idx].numRectangles-1].w = wnew;
		configs[idx].rectangles[configs[idx].numRectangles-1].c = cnew;
		configs[idx].rectangles[configs[idx].numRectangles-1].h = hnew;

		//Change height of other rect
		float w = configs[idx].rectangles[changes[idx].rect1].w;
		configs[idx].rectangles[changes[idx].rect1].h = (oldWeight - eps[idx]) / w;

	}


	/* Adds width to a rectangle in a configuration */
	__device__ void addWidth(elementaryChange* changes,  configuration* configs,  float* eps)
	{
		int idx = blockIdx.x;

		float weight = configs[idx].rectangles[changes[idx].rect1].w*configs[idx].rectangles[changes[idx].rect1].h;
		configs[idx].rectangles[changes[idx].rect1].w += eps[idx];
		configs[idx].rectangles[changes[idx].rect1].h = weight/configs[idx].rectangles[changes[idx].rect1].w;

	}

	/* Changes the weight of two rectangles in a configuration */
	__device__ void addWeight(elementaryChange* changes,  configuration* configs,  float* eps)
	{
		int idx = blockIdx.x;
		float w = configs[idx].rectangles[changes[idx].rect1].w;
		float wt = configs[idx].rectangles[changes[idx].rect2].w;
		configs[idx].rectangles[changes[idx].rect1].h += eps[idx];
		configs[idx].rectangles[changes[idx].rect2].h -= eps[idx]*w/wt;
	}

	/* Removes a rectangle in a configuration */
	__device__ void addRemove(elementaryChange* changes,  configuration* configs,  float* eps)
	{

		int idx = blockIdx.x;
		int r1 = changes[idx].rect1;
		int r2 = changes[idx].rect2;
		int numRectangles = configs[idx].numRectangles;

		float weight = configs[idx].rectangles[r1].w*configs[idx].rectangles[r1].h;
		configs[idx].rectangles[r2].h += weight/configs[idx].rectangles[r2].w;

		//Move the last rectangle to the removed slot
		if (r1 != (numRectangles - 1) )
		{
			configs[idx].rectangles[r1].w = configs[idx].rectangles[numRectangles - 1].w;
			configs[idx].rectangles[r1].c = configs[idx].rectangles[numRectangles - 1].c;
			configs[idx].rectangles[r1].h = configs[idx].rectangles[numRectangles - 1].h;
		}
		configs[idx].numRectangles--;

		changes[idx].rect1 = (r2 == numRectangles) ? r1 : r2;
		addShift(changes, configs, eps);

		//reset
		changes[idx].rect1 = r1;
		changes[idx].rect2 = r2;
	}

	/* Splits a rectangle in a configuration */
	__device__ void addSplit(elementaryChange* changes, configuration* configs, float* eps)
	{
		int idx = blockIdx.x;
		float w = configs[idx].rectangles[changes[idx].rect1].w;
		float h = configs[idx].rectangles[changes[idx].rect1].h;
		float c = configs[idx].rectangles[changes[idx].rect1].c;
		float r1 = changes[idx].rect1;
		float w1 = D_WIDTH_MIN + changes[idx].r*(w - D_WIDTH_MIN);
		float w2 = w - w1;

		//if split is not possible
		if(w <= 2*D_WIDTH_MIN || w1*h < D_WEIGHT_MIN || w2*h < D_WEIGHT_MIN || w1 < 0 || w2 < 0)
		{
			return;
		}

		configs[idx].rectangles[changes[idx].rect1].w = w1;
		configs[idx].rectangles[configs[idx].numRectangles].w = w2;
		configs[idx].rectangles[configs[idx].numRectangles].h = h;
		configs[idx].rectangles[configs[idx].numRectangles].c = c;

		configs[idx].numRectangles++;

		addShift(changes, configs, eps);
		eps[idx] = -eps[idx];
		changes[idx].rect1 = configs[idx].numRectangles - 1;
		addShift(changes, configs, eps);

		//Reset the change struct
		changes[idx].type = SPLIT;
		eps[idx] = -eps[idx]; 
		changes[idx].rect1 = r1;

	}

	/* Glues two rectangles in a configuration */
	__device__ void addGlue(elementaryChange* changes,  configuration* configs, float* eps)
	{
		int idx = blockIdx.x;
		int r1 = changes[idx].rect1;
		int r2 = changes[idx].rect2;
		float w = configs[idx].rectangles[r1].w;
		float wt = configs[idx].rectangles[r2].w;
		float h = configs[idx].rectangles[r1].h;
		float ht = configs[idx].rectangles[r2].h;
		float wnew = (w + wt)/2;
		float weight = w*h + wt*ht;
		float hnew = weight/wnew;
		float cnew = (configs[idx].rectangles[changes[idx].rect1].c + configs[idx].rectangles[changes[idx].rect2].c)/2;

		int savedIndex = r2 < r1 ? r2 : r1;
		int removedIndex = r2 > r1 ? r2 : r1;
		int numRectangles = configs[idx].numRectangles;

		//Remove the largest index (move last)
		if(removedIndex != (numRectangles - 1))
		{
			configs[idx].rectangles[removedIndex].w = configs[idx].rectangles[numRectangles - 1].w;
			configs[idx].rectangles[removedIndex].c = configs[idx].rectangles[numRectangles - 1].c;
			configs[idx].rectangles[removedIndex].h = configs[idx].rectangles[numRectangles - 1].h;
		}
		configs[idx].numRectangles--;

		//Set the new rectangle as the lowest index
		configs[idx].rectangles[savedIndex].w = wnew;
		configs[idx].rectangles[savedIndex].c = cnew;
		configs[idx].rectangles[savedIndex].h = hnew;

		//Shift the lowest index
		changes[idx].rect1 = savedIndex;
		addShift(changes, configs, eps);
		changes[idx].rect1 = r1; //reset
	}

	/* Prints an array of configurations */
	void printConfigs(configuration* configs, configuration* h_temp)
	{
		//Print rectangle data from current config
		rectangle rects[H_MAX_RECTANGLES];
		configuration confs[H_GRID_SIZE];
		cudaMemcpy(confs, configs, sizeof(configuration)*H_GRID_SIZE, cudaMemcpyDeviceToHost);
		for(int i = 0; i < H_GRID_SIZE; i++)
		{
			cudaMemcpy(rects, h_temp[i].rectangles, sizeof(rectangle)*H_MAX_RECTANGLES, cudaMemcpyDeviceToHost);
			printf("GRID %d (N: %d): \n", i, confs[i].numRectangles);
			for(int j = 0; j < confs[i].numRectangles; j++)
			{
				printf("whc (%f, %f, %f), \n", rects[j].w, rects[j].h, rects[j].c);
			}
			printf("\n");
		}
	}

	/* Update a configuration with type of change as input */
	__device__ void updateConfig(configuration* current, 
			elementaryChange* change, float* eps)
	{
		int idx = blockIdx.x;

		if(eps[idx] == D_EPS_ERROR)
		{
			return;
		}

		int type = change[idx].type;
		switch(type) 
		{
			case SHIFT:
				addShift(change, current, eps);
				break;
			case ADD:
				addRectangle(change, current, eps);
				break;
			case WIDTH:
				addWidth(change, current, eps);
				break;
			case WEIGHT:
				addWeight(change, current, eps);
				break;
			case REMOVE:
				addRemove(change, current, eps);
				break;
			case SPLIT:
				addSplit(change, current, eps);
				break;
			case GLUE:
				addGlue(change, current, eps);
				break;
		}  
	}

	/* Checks that a configuration holds all domain constraints */
	__global__ void checkConstraints(configuration* config, elementaryChange* change)
	{
		int idx = blockIdx.x;

		for(int i = 0; i < config[idx].numRectangles; i++)
		{
			if(config[idx].rectangles[i].w < 0)
			{
				printf("w < 0 from %d \n", change[idx].type);
			}
			if(config[idx].rectangles[i].h < 0)
			{
				printf("h < 0 from %d \n", change[idx].type);
			}
			if(config[idx].rectangles[i].c < D_OMEGA_MIN || config[idx].rectangles[i].c > D_OMEGA_MAX)
			{
				printf("c %f out of bounds from %d \n", config[idx].rectangles[i].c, change[idx].type);
			}
			if(config[idx].rectangles[i].w*config[idx].rectangles[i].h < D_WEIGHT_MIN)
			{
				printf("w*h < Smin from %d \n", change[idx].type);
			}
		}
	}

	/* Calculates the total weight for a all configurations */
	__global__ void totalWeight(configuration* config, double* weight)
	{
		int idx = blockIdx.x;
		weight[idx] = 0;
		float sum = 0;
		for(int i = 0 ; i < config[idx].numRectangles; i++)
		{
			sum += config[idx].rectangles[i].w * config[idx].rectangles[i].h;
		}
		weight[idx] = sum;
	}

	/* Creates all initial configurations */
	__global__ void createRandomConfiguration(configuration* currentConfig, curandState* states, int numRectangles)
	{

		int idx = blockIdx.x;

		float wh = 1.0f/numRectangles;
		for(int i = 0; i < numRectangles; i++)
		{
			float w = D_WIDTH_MIN + curand_uniform(&states[idx])*(1 - D_WIDTH_MIN);
			currentConfig[idx].rectangles[i].w = w;
			currentConfig[idx].rectangles[i].h = wh/w;
			float c = D_OMEGA_MIN + w/2 + curand_uniform(&states[idx])*(D_OMEGA_MAX - w/2 - (D_OMEGA_MIN + w/2));
			currentConfig[idx].rectangles[i].c = c;
		}
		currentConfig[idx].numRectangles = numRectangles;
	}

	/*
	 * Generates a proposed elementary update change for all global configurations
	 * The resulting elementary changes is stored in "changes"
	 * "states" is a curandState object for each global configuration
	 * "configs" is the configurations the changes are proposed to
	 * "eps1" is the output size of the change, (sometimes named delta ksi)
	 * "eps2" is half of "eps1"
	 **/
	__device__ void generateRandomChange(elementaryChange* changes, 
			curandState* states, configuration* configs,
			float* eps1, float* eps2)
	{
		int idx = blockIdx.x;
		float v = curand_uniform(&states[idx]);
		float r = curand_uniform(&states[idx]);

		//Will shift if numrectangles is max and tries to add or split a rectangle.  
		//Will also shift if numRectangles < 3 
		if(v < 0.143 && configs[idx].numRectangles < D_MAX_RECTANGLES) 
		{ 
			//Adding a rectangle
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			float weightmax = configs[idx].rectangles[i].h*configs[idx].rectangles[i].w;
			float a = curand_uniform(&states[idx]);
			changes[idx].r = a;
			changes[idx].emin = D_WEIGHT_MIN;
			changes[idx].emax = weightmax - D_WEIGHT_MIN;
			changes[idx].type = ADD;    
			changes[idx].rect1 = i;

		}
		else if(v < 0.143*2 && configs[idx].numRectangles < D_MAX_RECTANGLES)
		{
			//Splitting a rectangle
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			float r2 = curand_uniform(&states[idx]);
			changes[idx].rect1 = i;
			changes[idx].r = r2;
			changes[idx].type = SPLIT;

			float om = D_WIDTH_MIN + r2*(configs[idx].rectangles[i].w - D_WIDTH_MIN); //new width 1
			float om2 = configs[idx].rectangles[i].w - om; //new width 2

			float ws = om > om2 ? om : om2; //Select largest weight for omega   
			float c = configs[idx].rectangles[i].c;
			float emin = D_OMEGA_MIN - (c - ws/2);
			float emax = D_OMEGA_MAX - (c + ws/2);
			float ep = min(abs(emin), abs(emax));
			changes[idx].emin = -ep;
			changes[idx].emax = ep;
		} 
		else if(v < 0.143*3 && configs[idx].numRectangles > 2)
		{ 
			//Removing a rectangle
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			int i2 = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			if(i2 == i)
			{
				if(i2 == 0)
				{
					i2++;
				}else
				{
					i2--;
				}
			}
			float c = configs[idx].rectangles[i2].c;
			changes[idx].emin = D_OMEGA_MIN - (c - configs[idx].rectangles[i2].w/2);
			changes[idx].emax = D_OMEGA_MAX - (c + configs[idx].rectangles[i2].w/2);

			changes[idx].type = REMOVE;
			changes[idx].rect1 = i;
			changes[idx].rect2 = i2;

		} 
		else if(v < 0.143*4 && configs[idx].numRectangles > 2)
		{ 
			//Gluing a rectangle
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			int i2 = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			if(i2 == i)
			{
				if(i2 == 0)
				{
					i2++;
				}else
				{
					i2--;
				}
			}
			float wn = (configs[idx].rectangles[i].w + configs[idx].rectangles[i2].w)/2;
			changes[idx].type = GLUE;
			changes[idx].rect1 = i;
			changes[idx].rect2 = i2;
			float c = (configs[idx].rectangles[i].c + configs[idx].rectangles[i2].c)/2;
			changes[idx].emin = D_OMEGA_MIN - (c - wn/2);
			changes[idx].emax = D_OMEGA_MAX - (c + wn/2);
		}
		else if(v < 0.143*5)
		{ 
			//Shifting a rectangle
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			float c = configs[idx].rectangles[i].c;
			changes[idx].rect1 = i;
			changes[idx].type = SHIFT;
			changes[idx].emin = D_OMEGA_MIN - (c - configs[idx].rectangles[i].w/2);
			changes[idx].emax = D_OMEGA_MAX - (c + configs[idx].rectangles[i].w/2);
		} 
		else if(v < 0.143*6)
		{ 
			//Changing width of two rectangles
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			float c = configs[idx].rectangles[i].c;
			float w = configs[idx].rectangles[i].w;
			changes[idx].rect1 = i;
			changes[idx].type = WIDTH;
			changes[idx].emin = D_WIDTH_MIN - w/2;
			changes[idx].emax = min(2*(c - w/2 - D_OMEGA_MIN), 2*(D_OMEGA_MAX - c - w/2));
		} 
		else
		{ 
			//Changing the weight of two rectangles
			int i = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle
			int i2 = (int)(configs[idx].numRectangles*curand_uniform(&states[idx])); //random rectangle2
			if(i2 == i)
			{
				if(i2 == 0)
				{
					i2++;
				}
				else
				{
					i2--;
				}
			}
			float w = configs[idx].rectangles[i].w;
			float wt = configs[idx].rectangles[i2].w;
			float h = configs[idx].rectangles[i].h;
			float ht = configs[idx].rectangles[i2].h;

			changes[idx].rect1 = i;
			changes[idx].rect2 = i2;
			changes[idx].emin = max(D_WEIGHT_MIN/w - h, ht*wt/w - 1/w);
			changes[idx].emax = min(1/w - h, ht*wt/w - D_WEIGHT_MIN/w);
			changes[idx].type = WEIGHT;
		}

		eps1[idx] = changes[idx].emin + r*(changes[idx].emax - changes[idx].emin);
		eps2[idx] = eps1[idx]/2;
	}
} //end namespace
