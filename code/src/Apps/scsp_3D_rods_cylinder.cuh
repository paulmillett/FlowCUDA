
# ifndef SCSP_3D_RODS_CYLINDER_H
# define SCSP_3D_RODS_CYLINDER_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "../IBM/3D/class_rods_ibm3D.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_rods_cylinder : public FlowBase {
	
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
	int nRods;
	int nStepsEquilibrate;
	int nVTKOutputs;
	int iskip,jskip,kskip;
	int precision;
	float tau;
	float nu;
	float Pe;
	float umax;
	float bodyForx;
	float L0;
	float gam;
	float Lrod;
	float Drod;
	float chRad;
	float Q0;
	bool initRandom;
	
	// objects:
	class_scsp_D3Q19 lbm;
	class_rods_ibm3D rods;
		
public:

	scsp_3D_rods_cylinder();
	~scsp_3D_rods_cylinder();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_RODS_CYLINDER_H