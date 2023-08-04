
# ifndef SCSP_3D_CAPSULES_CONSTRICTION_H
# define SCSP_3D_CAPSULES_CONSTRICTION_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsule_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsules_constriction : public FlowBase {
	
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
	int nEdges;
	int nSteps;
	int iskip,jskip,kskip;
	float tau;
	float nu;
	float bodyForx;
	std::string vtkFormat;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsule_ibm3D ibm;
		
public:

	scsp_3D_capsules_constriction();
	~scsp_3D_capsules_constriction();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_CAPSULES_CONSTRICTION_H