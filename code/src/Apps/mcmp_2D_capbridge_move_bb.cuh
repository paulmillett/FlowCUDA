
# ifndef MCMP_2D_CAPBRIDGE_MOVE_BB_H
# define MCMP_2D_CAPBRIDGE_MOVE_BB_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/mcmp_SC_bb/class_mcmp_SC_bb_D2Q9.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_capbridge_move_bb : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nParts;
	float tau;
	float nu;
	float gAB;
	float gAS;
	float gBS;
	float pvel;
	float Khertz;
	float halo;
	std::string vtkFormat;
	
	// objects
	class_mcmp_SC_bb_D2Q9 lbm;
		
public:

	mcmp_2D_capbridge_move_bb();
	~mcmp_2D_capbridge_move_bb();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);

};

# endif  // MCMP_2D_CAPBRIDGE_MOVE_BB_H