#ifndef __CONSTANTS_CUH__
#define __CONSTANTS_CUH__

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/complex.h>


//The minimum width of a rectangle in a configuration
#define D_WIDTH_MIN 1/5001.0f

//The minimum weight of a rectangle in a configuration
#define D_WEIGHT_MIN 1/5001.0f

//A number indicator which describes when there is some error
//and the elementary update can not be used
#define D_EPS_ERROR 100000 

/*  Various constants, most are read from "control.in" */
namespace mish
{
	extern int H_MAX_RECTANGLES;
	extern int H_DENSITY_FUNCTION_POINTS;
	extern int H_GREENS_FUNCTION_POINTS;
	extern int H_GRID_SIZE;
	extern int H_BLOCK_SIZE;
	extern float H_OMEGA_MIN;
	extern float H_OMEGA_MAX;
	extern int H_LOCAL_UPDATES;
	extern int H_ELEMENTARY_UPDATES;
	extern int H_STARTING_RECTANGLES;
	extern int H_KERNEL;

	extern __constant__ int D_MAX_RECTANGLES;
	extern __constant__ int D_DENSITY_FUNCTION_POINTS;
	extern __constant__ int D_GREENS_FUNCTION_POINTS;
	extern __constant__ int D_GRID_SIZE;
	extern __constant__ int D_BLOCK_SIZE;
	extern __constant__ int D_LOCAL_UPDATES;
	extern __constant__ int D_ELEMENTARY_UPDATES;
	extern __constant__ float D_OMEGA_MIN;
	extern __constant__ float D_OMEGA_MAX;
	extern __constant__ float D_BETA;
	extern __constant__ int D_KERNEL;

} //end namespace
#endif


