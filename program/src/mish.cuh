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


#ifndef __MISH_CUH__
#define __MISH_CUH__

namespace mish
{

	//The input struct to the compute function, all values must be declared.
	struct input {
		float* input_greens_real; //Real parts of input Greens function (on the imaginary axis)
		float* input_greens_imag; //Imaginary parts of input Greens function (on the imaginary axis)
		float* input_args; 		  //Discrete mesh on which the Greens function is specified (matsubara frequencies or imaginary time)
		int num_input_points;     //The number of input Greens function points
		int num_global;			  //The number of global updates
		int num_local;			  //The number of local updates per global update
		int num_elementary;		  //The number of elementary updates per local update
		int kernel;               //Integer which specifies which kernel to use (see instruction file)
		float min_omega;		  //Minimum value of output energy range
		float max_omega;          //Maximum value of output energy range
		float beta;				  //Inverse temperature 
		int starting_rectangles;  //The number of rectangles in the starting configurations
		int num_output_points;    //The number of discrete points between min_omega and max_omega
	};

	/*
	 * Computes spectral function given a Green's function and a 
	 * kernel. The algorithm is described by Mischenko et.al in PRB 62 6317 (2000) appendix B3.
	 * The input data and parameters are stored in "inp".  
	 * The output spectral function is stored in "output".
	 * The output arguments, (I.e where the spectral function is defined is stored in "output_args".
	 */
	void compute(input &inp, float* output, float* output_args);

}  //end namespace

#endif
