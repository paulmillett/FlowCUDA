
# ifndef SCSP_3D_IOLETS_SOLIDWALLS_FORCING_H
# define SCSP_3D_IOLETS_SOLIDWALLS_FORCING_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_iolets_solidwalls_forcing : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int numIolets;
	float nu;
	float bodyForx;
	std::string vtkFormat;
	
	// scsp_D3Q19 object:
	class_scsp_D3Q19 lbm;
	
	
public:

	scsp_3D_iolets_solidwalls_forcing();
	~scsp_3D_iolets_solidwalls_forcing();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_IOLETS_SOLIDWALLS_FORCING_H