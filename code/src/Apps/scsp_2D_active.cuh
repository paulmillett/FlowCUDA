
# ifndef SCSP_2D_ACTIVE_H
# define SCSP_2D_ACTIVE_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/scsp_active/class_scsp_active_D2Q9.cuh"
# include <cuda.h>
# include <string>

class scsp_2D_active : public FlowBase {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nSteps;
	int iskip;
	int jskip;
	float nu;
	
	// objects:
	class_scsp_active_D2Q9 lbm;
	
public:

	scsp_2D_active();
	~scsp_2D_active();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	
};

# endif  // SCSP_2D_OBSTACLE_H