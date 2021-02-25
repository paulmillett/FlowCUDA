
# ifndef MCMP_3D_CAPBRIDGE_DIP_H
# define MCMP_3D_CAPBRIDGE_DIP_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/mcmp_SC_dip/class_mcmp_SC_dip_D3Q19.cuh"
# include <cuda.h>
# include <string>



class mcmp_3D_capbridge_dip : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nParts;
	float tau;
	float nu;
	float rApart;
	float rBpart;
	std::string vtkFormat;
	
	// objects
	class_mcmp_SC_dip_D3Q19 lbm;
	
		
public:

	mcmp_3D_capbridge_dip();
	~mcmp_3D_capbridge_dip();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);

};

# endif  // MCMP_3D_CAPBRIDGE_DIP_H