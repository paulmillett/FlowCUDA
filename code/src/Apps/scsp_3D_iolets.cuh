
# ifndef SCSP_3D_IOLETS_H
# define SCSP_3D_IOLETS_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_iolets : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int numIolets;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// host arrays:
	float* uH;
	float* vH;
	float* wH;
	float* rH;
	float* pH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	int* xH;
	int* yH;
	int* zH;
	iolet* ioletsH;
	
	// device arrays:
	float* u;
	float* v;
	float* w;
	float* r;
	float* p;
	float* f1;
	float* f2;
	int* voxelType;
	int* streamIndex;
	iolet* iolets;
	
public:

	scsp_3D_iolets();
	~scsp_3D_iolets();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_IOLETS_H