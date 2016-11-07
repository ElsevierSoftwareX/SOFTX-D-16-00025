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

/* main.cu
 * This program calculates an approximated spectral function given a
 * greens function and a kernel with GPU acceleration.
 * It is based on the algorithm by Mischenko et. al in PRB 62 6317 (2000) appendix B3.
 * The input greens function must be stored in "infloa.in"
 * The input parameters must be stored in "control.in" 
 * The output function will be stored in "out.dat"
 * */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mish.cuh"

/*
 Reads the constants from "control.in" and assigns them
 to the corresponding constants in inp 
 */
void readConstants(mish::input &inp) 
{
	FILE *infloa;
	char line[80];

	infloa = fopen("infloa.in","r");
	if(infloa == NULL)
	{
		perror("Error reading infloa");
		exit(EXIT_FAILURE);
	}

	//Read number of input points
	int i = 0;
	while(fgets(line, 80, infloa) != NULL)
	{
		i++;
	}
	inp.num_input_points = i;

	infloa = fopen("control.in","r");
	if(infloa == NULL)
	{
		perror("Error reading input");
		exit(EXIT_FAILURE);
	}
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.num_global);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.num_local);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.num_elementary);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.kernel);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%f", &inp.min_omega);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%f", &inp.max_omega);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%f", &inp.beta);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.starting_rectangles);
	if(fgets(line, 80, infloa) == NULL) 
	{
		printf("Error reading control.in \n");
	}
	sscanf(line, "%d", &inp.num_output_points);

	fclose(infloa);

}


/*
 Reads the input Green's function from "infloa.in"
 and copies to the inp._input_greens.
 */
void readGreensFunction(mish::input &inp, float* real, float* imag, float* args)
{
	FILE *infloa;
	char line[80];

	infloa = fopen("infloa.in","r");
	if(infloa == NULL)
	{
		perror("Error reading infloa");
		exit(EXIT_FAILURE);
	}

	int i = 0;
	while(fgets(line, 80, infloa) != NULL)
	{
		float im = 0;
		int num = sscanf(line, "%f %f %f", &args[i], &real[i], &im);
		imag[i] = im;
		if(num < 2){
			perror("Error reading infloa");
			exit(EXIT_FAILURE);
		}
		i++;
	}
	inp.input_greens_real = real;
	inp.input_greens_imag = imag;
	inp.input_args = args;
	inp.num_input_points = i;
	fclose(infloa);
}

/*
 * Writes the final output to "out.dat"
 * */
void writeToFile(float* output, float* output_args, int num)
{
	FILE *out;
	out = fopen("out.dat","w");
	if(out == NULL)
	{
		perror("Error opening out.dat");
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < num; i++)
	{
		fprintf(out, "%f %f\n", output_args[i], output[i]);
	}
	fclose(out);
}

/*
 * This program approximates a spectral function given a Green's function and a 
 * kernel. The algorithm is described by Mischenko et.al in PRB 62 6317 (2000) appendix B3.
 * */
int main() 
{

	//The input parameters are stored in inp
	mish::input inp;

	//Read input parameters from "control.in"
	readConstants(inp);
	
	//Allocate the input and output float arrays
	float* real = (float*)malloc(sizeof(float)*inp.num_input_points); //The real part of the input Greens function
	float* imag = (float*)malloc(sizeof(float)*inp.num_input_points); //The imaginary part of the input Greens function
	float* args = (float*)malloc(sizeof(float)*inp.num_input_points); //Discrete mesh on which the Greens function is specified (matsubara frequencies or imaginary time)
	float* output = (float*)malloc(sizeof(float)*inp.num_output_points); //The output buffer where final spectral function will be stored
	float* output_args = (float*)malloc(sizeof(float)*inp.num_output_points); //The output argument (discrete energy mesh) buffer 

	//Read the input greens function "infloa.in"
	readGreensFunction(inp, real, imag, args);

	//Compute the spectral function on the gpu
	mish::compute(inp, output, output_args);

	//Write to out.dat
	writeToFile(output, output_args, inp.num_output_points);

	//Free the bufers
	free(real);
	free(imag);
	free(args);
	free(output);
	free(output_args);

	return EXIT_SUCCESS;
}

