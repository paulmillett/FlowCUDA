
# ifndef SCSP_3D_CAPSULES_DUCT_MARGINATION_H
# define SCSP_3D_CAPSULES_DUCT_MARGINATION_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsules_binary_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsules_duct_margination : public FlowBase {
	
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
	float Re;
	float Ca1;
	float Ca2;
	float tau;
	float nu;
	float bodyForx;
	float umax;
	float a1;
	float a2;
	float gam;
	float Q0;
	float trainRij;
	float trainAng;
	bool initRandom;
	std::string vtkFormat;
	std::string ibmUpdate;
	std::string ibmFile1;
	std::string ibmFile2;
	std::string cellProps;
	std::string cellSizes;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_capsules_binary_ibm3D ibm;
		
public:

	scsp_3D_capsules_duct_margination();
	~scsp_3D_capsules_duct_margination();
	void initSystem();
	void cycleForward(int,int);
	void stepIBM();
	void stepVerlet();
	void writeOutput(std::string,int);
	void calcMembraneParams();
	float calcInfSum(float,float);
	
};

# endif  // SCSP_3D_CAPSULES_DUCT_MARGINATION_H