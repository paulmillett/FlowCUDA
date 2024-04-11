
# ifndef SCSP_2D_ACTIVE_DROPLET_2PHI_H
# define SCSP_2D_ACTIVE_DROPLET_2PHI_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/scsp_active/class_scsp_active_2phi_D2Q9.cuh"
# include <cuda.h>
# include <string>

class scsp_2D_active_droplet_2phi : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nSteps;
	float nu;
	float dropRad;
	
	// objects:
	class_scsp_active_2phi_D2Q9 lbm;
	
public:

	scsp_2D_active_droplet_2phi();
	~scsp_2D_active_droplet_2phi();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_2D_ACTIVE_DROPLET_2PHI_H