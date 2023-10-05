
# ifndef SCSP_3D_CAPSULE_VISC_CONTRAST_H
# define SCSP_3D_CAPSULE_VISC_CONTRAST_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsules_ibm3D.cuh"
# include "../IBM/3D/class_poisson_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsule_visc_contrast : public FlowBase {
	
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
	int nCells;
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	int precision;
	float tau;
	float nu;
	float gam;
	float shearVel;
	float a;
	float nu_in;
	float nu_out;
	bool initRandom;
	std::string vtkFormat;
	std::string ibmFile;
	std::string ibmUpdate;
	std::string cellProps;
	std::string cellSizes;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsules_ibm3D ibm;
	class_poisson_ibm3D poisson;
		
public:

	scsp_3D_capsule_visc_contrast();
	~scsp_3D_capsule_visc_contrast();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	void calcMembraneParams(float,float);
	
};

# endif  // SCSP_3D_CAPSULE_VISC_CONTRAST_H