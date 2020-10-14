
# ifndef SCSP_2D_EXPAND_2_H
# define SCSP_2D_EXPAND_2_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_2D_expand_2 : public FlowBase {
	
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
	int nSteps;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// host arrays:
	float* uH;
	float* vH;
	float* rH;
	float* xIBH;
	float* yIBH;
	float* xIBH_start;
	float* yIBH_start;
	float* xIBH_end;
	float* yIBH_end;
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
	float* xIB_start;
	float* yIB_start;
	float* xIB_end;
	float* yIB_end;
	float* vxIB;
	float* vyIB;
	float* fxIB;
	float* fyIB;
	float* uIBvox;
	float* vIBvox;
	float* weights;
	int* voxelType;
	int* streamIndex;
	iolet2D* iolets;
	
public:

	scsp_2D_expand_2();
	~scsp_2D_expand_2();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_2D_EXPAND_2_H