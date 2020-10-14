
# ifndef SCSP_3D_IOLETS_2_H
# define SCSP_3D_IOLETS_2_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/classes/scsp_D3Q19.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_iolets_2 : public FlowBase {
	
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
	
	// scsp_D3Q19 object:
	scsp_D3Q19 lbm;
	
	
public:

	scsp_3D_iolets_2();
	~scsp_3D_iolets_2();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_IOLETS_2_H