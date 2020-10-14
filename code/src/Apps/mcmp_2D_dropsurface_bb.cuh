
# ifndef MCMP_2D_DROPSURFACE_BB_H
# define MCMP_2D_DROPSURFACE_BB_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/particles/particle_struct_D2Q9.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_dropsurface_bb : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nParticles;
	float tau;
	float nu;
	float gAB;
	float gAS;
	float gBS;
	std::string vtkFormat;
		
	// host arrays:
	float* uH;
	float* vH;
	float* rAH;
	float* rBH;
	int* sH;
	int* xH;
	int* yH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	int* pIDH;
	particle2D* pH;
	
	// device arrays:
	float* u;
	float* v;
	float* rA;
	float* rB;
	float* f1A;
	float* f1B;
	float* f2A;
	float* f2B;
	float* FxA;
	float* FxB;
	float* FyA;
	float* FyB;
	int* x;
	int* y;
	int* s;
	int* nList;
	int* voxelType;
	int* streamIndex;
	int* pID;
	particle2D* p;
	
public:

	mcmp_2D_dropsurface_bb();
	~mcmp_2D_dropsurface_bb();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	int  voxelIndex(int,int,int);

};

# endif  // MCMP_2D_DROPSURFACE_BB_H