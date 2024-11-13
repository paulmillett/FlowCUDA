
# ifndef SCSP_3D_CAPSULES_DUCT_TRAINS_PUSH_H
# define SCSP_3D_CAPSULES_DUCT_TRAINS_PUSH_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsules_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsules_duct_trains_push : public FlowBase {
	
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
	float bodyForx;
	float umax;
	float a;
	float gam;
	float Q0;
	float trainRij;
	float trainAng;
	float wavelength;
	bool initRandom;
	bool pulsatile;
	std::string vtkFormat;
	std::string ibmUpdate;
	std::string ibmFile;
	std::string cellProps;
	std::string cellSizes;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsules_ibm3D ibm;
		
public:

	scsp_3D_capsules_duct_trains_push();
	~scsp_3D_capsules_duct_trains_push();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	void calcMembraneParams(float,float,float);
	void calcRefFlux();
	float calcInfSum(float,float);
	
};

# endif  // SCSP_3D_CAPSULES_DUCT_TRAINS_PUSH_H