
# ifndef SCSP_3D_HEMISPHERE_2_H
# define SCSP_3D_HEMISPHERE_2_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_hemisphere_2 : public FlowBase {
	
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
	int nFaces;
	int nSteps;
	int iskip,jskip,kskip;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// host arrays:
	float* uH;
	float* vH;
	float* wH;
	float* rH;
	float* xIBH;
	float* yIBH;
	float* zIBH;
	float* xIBH_start;
	float* yIBH_start;
	float* zIBH_start;
	float* xIBH_end;
	float* yIBH_end;
	float* zIBH_end;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	int* inoutH;
	int* xH;
	int* yH;
	int* zH;
	int* faceV1;
	int* faceV2;
	int* faceV3;
	iolet* ioletsH;
	
	// device arrays:
	float* u;
	float* v;
	float* w;
	float* r;
	float* f1;
	float* f2;
	float* Fx;
	float* Fy;
	float* Fz;
	float* uIBvox;
	float* vIBvox;
	float* wIBvox;
	float* weights;
	float* xIB;
	float* yIB;
	float* zIB;
	float* xIB_start;
	float* yIB_start;
	float* zIB_start;
	float* xIB_end;
	float* yIB_end;
	float* zIB_end;
	float* vxIB;
	float* vyIB;
	float* vzIB;
	float* fxIB;
	float* fyIB;
	float* fzIB;
	int* voxelType;
	int* streamIndex;
	int* inout;
	iolet* iolets;
	
public:

	scsp_3D_hemisphere_2();
	~scsp_3D_hemisphere_2();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_HEMISPHERE_2_H