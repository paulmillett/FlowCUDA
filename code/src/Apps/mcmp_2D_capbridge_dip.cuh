
# ifndef MCMP_2D_CAPBRIDGE_DIP_H
# define MCMP_2D_CAPBRIDGE_DIP_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/mcmp_SC_dip/class_mcmp_SC_dip_D2Q9.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_capbridge_dip : public FlowBase {
	
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
	class_mcmp_SC_dip_D2Q9 lbm;
	
		
public:

	mcmp_2D_capbridge_dip();
	~mcmp_2D_capbridge_dip();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);

};

# endif  // MCMP_2D_CAPBRIDGE_DIP_H