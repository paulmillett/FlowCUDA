
# ifndef CLASS_POISSON_IBM3D_H
# define CLASS_POISSON_IBM3D_H

# include "../../Utils/helper_math.h"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_poisson_ibm3D.cuh"
# include "membrane_data.h"
# include <cuda.h>
# include <cufft.h>
# include <string>


class class_poisson_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int Nx;
	int Ny;
	int Nz;
	int nVoxels;
	cufftHandle plan;
	
	// host arrays:
	float* indicatorH;
				
	// device arrays:
	float* indicator;
	float* kx;
	float* ky;
	float* kz;
	float3* G;
	cufftComplex* rhs;
	
	// methods:
	class_poisson_ibm3D();
	~class_poisson_ibm3D();
	void initialize(int,int,int);
	void solve_poisson(triangle*,float3*,int,int,int);
	void write_output(std::string,int,int,int,int,int);
	void deallocate();	
	
};

# endif  // CLASS_POISSON_IBM3D_H