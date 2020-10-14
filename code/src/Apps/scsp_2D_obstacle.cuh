
# ifndef SCSP_2D_OBSTACLE_H
# define SCSP_2D_OBSTACLE_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_2D_obstacle : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int nBlocksIB;
	int Nx,Ny,Nz;
	int numIolets;
	int nNodes;
	float tau;
	float nu;
	float kstiff;
	std::string vtkFormat;
	
	// host arrays:
	float* uH;
	float* vH;
	float* rH;
	float* xIBH;
	float* yIBH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	int* xH;
	int* yH;
	iolet2D* ioletsH;
	
	// device arrays:
	float* u;
	float* v;
	float* r;
	float* f1;
	float* f2;
	float* Fx;
	float* Fy;
	float* xIB;
	float* yIB;
	float* xIB0;
	float* yIB0;
	float* vxIB;
	float* vyIB;
	float* fxIB;
	float* fyIB;
	int* voxelType;
	int* streamIndex;
	iolet2D* iolets;
	
public:

	scsp_2D_obstacle();
	~scsp_2D_obstacle();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_2D_OBSTACLE_H