
# ifndef SCSP_2D_EXPAND_H
# define SCSP_2D_EXPAND_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/iolets/boundary_condition_iolet.cuh"
# include "../D2Q9/scsp/class_scsp_D2Q9.cuh"
# include "../IBM/2D/class_ibm2D.cuh"
# include <cuda.h>
# include <string>

class scsp_2D_expand : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int nBlocksIB;
	int Nx,Ny,Nz;
	int numIolets;
	int nNodes;
	int nSteps;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// objects:
	class_scsp_D2Q9 lbm;
	class_ibm2D ibm;
		
public:

	scsp_2D_expand();
	~scsp_2D_expand();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_2D_EXPAND_H