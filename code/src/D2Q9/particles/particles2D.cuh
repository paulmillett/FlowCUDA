
# ifndef PARTICLES2D_H
# define PARTICLES2D_H

# include <cuda.h>
# include <string>

struct particles2D {
	
	// data:
	int nParts;
	int nVoxels; 
			
	// host arrays:
	float* xH;
	float* yH;
	float* radH;
			
	// device arrays:
	float* x;
	float* y;	
	float* vx;
	float* vy;
	float* fx;
	float* fy;
	float* rad;
	int* pIDgrid;
	
	// methods:
	particles2D();
	~particles2D();
	void allocate();
	void deallocate();		
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	
	void write_output(std::string,int);
	
};

# endif  // PARTICLES2D_H