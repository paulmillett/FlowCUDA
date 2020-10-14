
# ifndef MCMP_2D_BASIC_H
# define MCMP_2D_BASIC_H

# include "../Base/FlowBase.cuh"
# include <cuda.h>
# include <string>

class mcmp_2D_basic : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int potType;
	float tau;
	float nu;
	float gAB;
	std::string vtkFormat;
	
	// host arrays:
	float* uH;
	float* vH;
	float* rAH;
	float* rBH;
	float* prH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	
	// device arrays:
	float* u;
	float* v;
	float* rA;
	float* rB;
	float* pr;
	float* f1A;
	float* f1B;
	float* f2A;
	float* f2B;
	float* FxA;
	float* FxB;
	float* FyA;
	float* FyB;
	int* nList;
	int* voxelType;
	int* streamIndex;
	
public:

	mcmp_2D_basic();
	~mcmp_2D_basic();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	int  voxelIndex(int,int,int);

};

# endif  // MCMP_2D_BASIC_H