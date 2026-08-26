
# ifndef SCSP_3D_DISCS_SHEAR_H
# define SCSP_3D_DISCS_SHEAR_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_discs_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_discs_shear : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int nBlocksIB;
	int Nx,Ny,Nz;
	int numIolets;
	int nBeads;
	int nSteps;
	int nDiscs;
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	int precision;
	float tau;
	float nu;
	float shearVel;
	float gam;
	float Ddisc;
	float Hdisc;
	bool initRandom;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_discs_ibm3D discs;
		
public:

	scsp_3D_discs_shear();
	~scsp_3D_discs_shear();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_DISCS_SHEAR_H