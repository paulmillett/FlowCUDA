
# ifndef MCMP_3D_CAPBRIDGE_BB_H
# define MCMP_3D_CAPBRIDGE_BB_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/mcmp_SC_bb/class_mcmp_SC_bb_D3Q19.cuh"
# include <cuda.h>
# include <string>



class mcmp_3D_capbridge_bb : public FlowBase {
	
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
	float gAB;	
	float pvel;
	std::string vtkFormat;
	
	// objects
	class_mcmp_SC_bb_D3Q19 lbm;
		
public:

	mcmp_3D_capbridge_bb();
	~mcmp_3D_capbridge_bb();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);

};

# endif  // MCMP_3D_CAPBRIDGE_BB_H