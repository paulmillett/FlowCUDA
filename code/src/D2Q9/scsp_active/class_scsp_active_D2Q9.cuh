
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
	float a;
	float alpha;
	float beta;
	float kapp;
	float kapphi;
	float mob;
		
	// host arrays:
	float* rH;
	float* phiH;
	float2* uH;
	float2* pH;
	float2* hH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	
	// device arrays:	
	float* r;
	float* f1;
	float* f2;
	float* phi;
	float* chempot;
	float2* u;
	float2* F;
	float2* p;
	float2* h;
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
	void setPhi(int,float);
	void setVoxelType(int,int);
	float getU(int);
	float getV(int);
	float getR(int);
	float getPhi(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int);	
	void stream_collide_save_forcing(int,int);
	void set_wall_velocity_ydir(float,int,int);
	void scsp_active_update_orientation(int,int);
	void scsp_active_update_orientation_diffusive(int,int);
	void scsp_active_fluid_stress(int,int);
	void scsp_active_fluid_forces(int,int);
	void scsp_active_fluid_molecular_field(int,int);
	void scsp_active_fluid_molecular_field_with_phi(int,int);
	void scsp_active_fluid_chemical_potential(int,int);
	void scsp_active_fluid_capillary_force(int,int);
	void scsp_active_fluid_update_phi(int,int);
	void scsp_active_fluid_update_phi_diffusive(int,int);
	void scsp_active_fluid_set_velocity_field(int,int);
	void zero_forces(int,int);	
	void write_output(std::string,int,int,int);

};

# endif  // CLASS_SCSP_ACTIVE_D2Q9_H