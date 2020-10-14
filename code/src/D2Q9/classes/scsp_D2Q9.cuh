
# ifndef SCSP_D2Q9_H
# define SCSP_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include <cuda.h>
# include <string>

class scsp_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	float nu;
	bool forceFlag;
	bool velIBFlag;
		
	// host arrays:
	float* uH;
	float* vH;
	float* rH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	iolet2D* ioletsH;
	
	// device arrays:
	float* u;
	float* v;
	float* r;
	float* f1;
	float* f2;
	float* Fx;
	float* Fy;
	float* uIBvox;
	float* vIBvox;
	int* voxelType;
	int* streamIndex;
	iolet2D* iolets;
	
public:

	scsp_D2Q9();
	~scsp_D2Q9();
	void allocate();
	void deallocate();
	void allocate_forces();
	void allocate_IB_velocities();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_lattice_box();
	void create_lattice_box_periodic();
	void create_lattice_file();
	void stream_index_push();
	void stream_index_pull();
	void read_iolet_info(int,const char*);
	void setU(int,float);
	void setV(int,float);
	void setR(int,float);
	void setVoxelType(int,int);
	float getU(int);
	float getV(int);
	float getR(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int,bool);	
	void stream_collide_save_forcing(int,int);
	void zero_forces(int,int);	

};

# endif  // SCSP_D2Q9_H