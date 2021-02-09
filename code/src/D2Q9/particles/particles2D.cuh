
# ifndef PARTICLES2D_H
# define PARTICLES2D_H

# include "move_particles_2D.cuh"
# include "zero_forces_2D.cuh"
# include <cuda.h>
# include <string>

struct particles2D {
	
	// data:
	int nParts;
	int nVoxels;
	float rApart;
	float rBpart; 
			
	// host arrays:
	float* xH;
	float* yH;
	float* radH;
	float* rInnerH;
	float* rOuterH;
			
	// device arrays:
	float* x;
	float* y;	
	float* vx;
	float* vy;
	float* fx;
	float* fy;
	float* rad;
	float* rInner;
	float* rOuter;
	float* B;
	int* pIDgrid;
	
	// methods:
	particles2D();
	~particles2D();
	void allocate();
	void deallocate();		
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void zero_forces(int,int);
	void move_particles(int,int);
	void write_output(std::string,int);
	
};

# endif  // PARTICLES2D_H