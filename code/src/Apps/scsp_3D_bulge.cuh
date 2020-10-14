
# ifndef SCSP_3D_BULGE_H
# define SCSP_3D_BULGE_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/classes/scsp_D3Q19.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_bulge : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int numIolets;
	int nSteps;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// scsp_D3Q19 object:
	scsp_D3Q19 lbm;
	
	
public:

	scsp_3D_bulge();
	~scsp_3D_bulge();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_BULGE_H