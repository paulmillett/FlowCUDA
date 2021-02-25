
# ifndef MCMP_2D_DROPSURFACE_BB_H
# define MCMP_2D_DROPSURFACE_BB_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/particles/particle_struct_D2Q9.cuh"
# include "../D2Q9/mcmp_SC_bb/class_mcmp_SC_bb_D2Q9.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_dropsurface_bb : public FlowBase {
	
private:

	// scalars:
	int Q; 
	int nVoxels;	
	int nBlocks;
	int nThreads;
	int Nx,Ny,Nz;
	int nParticles;
	float tau;
	float nu;
	float gAB;
	float gAS;
	float gBS;
	std::string vtkFormat;
	
	// objects
	class_mcmp_SC_bb_D2Q9 lbm;
		
public:

	mcmp_2D_dropsurface_bb();
	~mcmp_2D_dropsurface_bb();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);
	int  voxelIndex(int,int,int);

};

# endif  // MCMP_2D_DROPSURFACE_BB_H