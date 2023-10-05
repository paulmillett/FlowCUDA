
# ifndef SCSP_3D_CAPSULE_SKALAK_VERLET_H
# define SCSP_3D_CAPSULE_SKALAK_VERLET_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsules_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsule_skalak_verlet : public FlowBase {
	
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
	float shearVel;
	float gam;
	std::string vtkFormat;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsules_ibm3D ibm;
		
public:

	scsp_3D_capsule_skalak_verlet();
	~scsp_3D_capsule_skalak_verlet();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_CAPSULE_SKALAK_VERLET_H