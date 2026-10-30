
# ifndef SCSP_3D_PSM_SHEAR_H
# define SCSP_3D_PSM_SHEAR_H

# include "../Base/FlowBase.cuh"
# include "../D3Q19/scsp_psm/class_scsp_psm_D3Q19.cuh"
# include <cuda.h>
# include <string>

class scsp_3D_psm_shear : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int numIolets;
	int nSpheres;
	int nSteps;
	int iskip,jskip,kskip;
	int precision;
	int nVTKOutputs;
	float tau;
	float nu;
	float shearVel;
	
	// objects:
	class_scsp_psm_D3Q19 lbm;
		
public:

	scsp_3D_psm_shear();
	~scsp_3D_psm_shear();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_3D_PSM_SHEAR_H