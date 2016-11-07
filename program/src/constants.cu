#include "constants.cuh"

namespace mish
{
	/* Various constants */
	__constant__ float D_OMEGA_MIN;
	__constant__ float D_OMEGA_MAX;
	__constant__ int D_MAX_RECTANGLES;
	__constant__ int D_DENSITY_FUNCTION_POINTS;
	__constant__ int D_GREENS_FUNCTION_POINTS;
	__constant__ int D_GRID_SIZE;
	__constant__ int D_BLOCK_SIZE;
	__constant__ int D_LOCAL_UPDATES;
	__constant__ int D_ELEMENTARY_UPDATES;
	__constant__ float D_BETA;
	__constant__ int D_KERNEL;

	int H_MAX_RECTANGLES;
	int H_DENSITY_FUNCTION_POINTS;
	int H_GREENS_FUNCTION_POINTS;
	int H_GRID_SIZE;
	int H_BLOCK_SIZE;
	float H_OMEGA_MIN;
	float H_OMEGA_MAX;
	int H_LOCAL_UPDATES;
	int H_ELEMENTARY_UPDATES;
	int H_STARTING_RECTANGLES;
	int H_KERNEL;
} //end namespace
