
# ifndef SCSP_3D_CAPSULES_DUCT_OVAL_MARGINATION_H
# define SCSP_3D_CAPSULES_DUCT_OVAL_MARGINATION_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_capsules_binary_ibm3D.cuh"
# include "../IBM/3D/class_poisson_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_capsules_duct_oval_margination : public FlowBase {
	
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
	float nu_in;
	float nu_out;
	float chA;
	float chB;
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
	class_poisson_ibm3D poissonRBC;
	class_poisson_ibm3D poissonPLT;
	
		
public:

	scsp_3D_capsules_duct_oval_margination();
	~scsp_3D_capsules_duct_oval_margination();
	void initSystem();
	void cycleForward(int,int);
	void stepIBM();
	void stepVerlet();
	void writeOutput(std::string,int);
	void calcMembraneParams();
	float calcInfSum(float,float);
	float wall_shear_rate(std::string);
	float velocity_at_point(float,float,float,float);
	void calcRefFlux();
	
};

# endif  // SCSP_3D_CAPSULES_DUCT_MARGINATION_H