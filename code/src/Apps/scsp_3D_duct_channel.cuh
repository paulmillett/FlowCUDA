
# ifndef SCSP_3D_DUCT_CHANNEL_H
# define SCSP_3D_DUCT_CHANNEL_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_duct_channel : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int nBlocksIB;
	int Nx,Ny,Nz;
	int numIolets;	
	int nSteps;
	int iskip,jskip,kskip;
	float tau;
	float nu;
	float bodyForx;
	std::string vtkFormat;
	
	// objects:
	class_scsp_D3Q19 lbm;
		
public:

	scsp_3D_duct_channel();
	~scsp_3D_duct_channel();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_DUCT_CHANNEL_H