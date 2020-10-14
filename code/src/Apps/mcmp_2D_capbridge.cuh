
# ifndef MCMP_2D_CAPBRIDGE_H
# define MCMP_2D_CAPBRIDGE_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/particles/particle_struct_D2Q9.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_capbridge : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nParticles;
	int potType;
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
	float* rSH;
	int* xH;
	int* yH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	particle2D* pH;
	
	// device arrays:
	float* u;
	float* v;
	float* rA;
	float* rB;
	float* rS;
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
	int* nList;
	int* voxelType;
	int* streamIndex;
	int* pID;
	particle2D* p;
	
public:

	mcmp_2D_capbridge();
	~mcmp_2D_capbridge();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	int  voxelIndex(int,int,int);

};

# endif  // MCMP_2D_CAPBRIDGE_H