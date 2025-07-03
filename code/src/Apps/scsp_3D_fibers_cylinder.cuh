
# ifndef SCSP_3D_FIBERS_CYLINDER_H
# define SCSP_3D_FIBERS_CYLINDER_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_fibers_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_fibers_cylinder : public FlowBase {
	
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
	int nFibers;
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	int precision;
	float chRad;
	float tau;
	float nu;
	float bodyForx;
	float umax;	
	float gam;
	float dS;
	float Lfib;
	bool initRandom;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_fibers_ibm3D fibers;
		
public:

	scsp_3D_fibers_cylinder();
	~scsp_3D_fibers_cylinder();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_FIBERS_CYLINDER_H