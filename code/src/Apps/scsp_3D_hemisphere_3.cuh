
# ifndef SCSP_3D_HEMISPHERE_3_H
# define SCSP_3D_HEMISPHERE_3_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/classes/scsp_D3Q19.cuh"
# include "../IBM/3D/struct_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_hemisphere_3 : public FlowBase {
	
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
	int nFaces;
	int nSteps;
	int iskip,jskip,kskip;
	float tau;
	float nu;
	std::string vtkFormat;
	
	// objects:
	scsp_D3Q19 lbm;
	struct_ibm3D ibm;
		
public:

	scsp_3D_hemisphere_3();
	~scsp_3D_hemisphere_3();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_HEMISPHERE_3_H