
# ifndef MCMP_2D_CAPBRIDGE_PSM_H
# define MCMP_2D_CAPBRIDGE_PSM_H

# include "../Base/FlowBase.cuh"
# include "../D2Q9/classes/mcmp_D2Q9.cuh"
# include "../D2Q9/particles/particles2D.cuh"
# include <cuda.h>
# include <string>



class mcmp_2D_capbridge_psm : public FlowBase {
	
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
	float rApart;
	float rBpart;
	std::string vtkFormat;
	
	// objects
	mcmp_D2Q9 lbm;
	particles2D parts;
		
public:

	mcmp_2D_capbridge_psm();
	~mcmp_2D_capbridge_psm();
	void initSystem();
	void cycleForward(int,int);
	void writeOutput(std::string,int);

};

# endif  // MCMP_2D_CAPBRIDGE_PSM_H