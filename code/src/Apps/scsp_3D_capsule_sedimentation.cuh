
# ifndef SCSP_3D_CAPSULE_SEDIMENTATION_H
# define SCSP_3D_CAPSULE_SEDIMENTATION_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsule_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsule_sedimentation : public FlowBase {
	
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
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	float tau;
	float nu;
	float a;
	float gam;
	float fx;
	std::string vtkFormat;
	std::string ibmUpdate;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsule_ibm3D ibm;
		
public:

	scsp_3D_capsule_sedimentation();
	~scsp_3D_capsule_sedimentation();
	void initSystem();
	void cycleForward(int,int);
	void stepIBM();
	void stepVerlet();
	void writeOutput(std::string,int);
	void calcMembraneParams(float,float,float,float);
	
};

# endif  // SCSP_3D_CAPSULE_SEDIMENTATION_H