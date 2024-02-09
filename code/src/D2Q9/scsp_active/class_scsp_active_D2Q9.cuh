
# ifndef CLASS_SCSP_ACTIVE_D2Q9_H
# define CLASS_SCSP_ACTIVE_D2Q9_H

# include "../init/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "tensor2D.h" 
# include <cuda.h>
# include <string>



class class_scsp_active_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	float nu;
	float sf;
	float fricR;
	float activity;
		
	// host arrays:
	float* rH;
	float2* uH;
	float2* pH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	
	// device arrays:	
	float* r;
	float* f1;
	float* f2;
	float2* u;
	float2* F;
	float2* p;
	tensor2D* stress;
	int* voxelType;
	int* streamIndex;
	int* nList;
	
public:

	class_scsp_active_D2Q9();
	~class_scsp_active_D2Q9();
	void allocate();
	void deallocate();
	void allocate_forces();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_lattice_box_periodic();
	void stream_index_pull();
	void setU(int,float);
	void setV(int,float);
	void setPx(int,float);
	void setPy(int,float);
	void setR(int,float);
	void setVoxelType(int,int);
	float getU(int);
	float getV(int);
	float getR(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int);	
	void stream_collide_save_forcing(int,int);
	void scsp_active_update_orientation(int,int);
	void scsp_active_fluid_stress(int,int);
	void scsp_active_fluid_forces(int,int);
	void zero_forces(int,int);	
	void write_output(std::string,int);

};

# endif  // CLASS_SCSP_ACTIVE_D2Q9_H