
# ifndef SCSP_3D_FILAMENTS_H
# define SCSP_3D_FILAMENTS_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_filaments_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_filaments : public FlowBase {
	
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
	int nEdges;
	int nSteps;
	int nFilams;
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	int precision;
	float tau;
	float nu;
	float shearVel;
	float ks,kb;
	float fp;
	float L0;
	float Lfil;
	bool initRandom;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_filaments_ibm3D filams;
		
public:

	scsp_3D_filaments();
	~scsp_3D_filaments();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_FILAMENTS_H