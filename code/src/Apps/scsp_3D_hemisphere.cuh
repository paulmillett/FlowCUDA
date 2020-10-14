
# ifndef SCSP_3D_HEMISPHERE_H
# define SCSP_3D_HEMISPHERE_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_hemisphere : public FlowBase {
	
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
	float kstiff;
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
	float* xIB;
	float* yIB;
	float* zIB;
	float* xIBref;
	float* yIBref;
	float* zIBref;
	float* xIBref_start;
	float* yIBref_start;
	float* zIBref_start;
	float* xIBref_end;
	float* yIBref_end;
	float* zIBref_end;
	float* vxIB;
	float* vyIB;
	float* vzIB;
	float* fxIB;
	float* fyIB;
	float* fzIB;
	int* voxelType;
	int* streamIndex;
	iolet* iolets;
	
public:

	scsp_3D_hemisphere();
	~scsp_3D_hemisphere();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_HEMISPHERE_H