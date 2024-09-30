
# ifndef CLASS_SCSP_ACTIVE_3PHI_D2Q9_H
# define CLASS_SCSP_ACTIVE_3PHI_D2Q9_H

# include "../init/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "tensor2D.h" 
# include <cuda.h>
# include <string>



class class_scsp_active_3phi_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	int stepprev;
	float nu;
	float nu_in;
	float nu_out;
	float sf;
	float fricR;
	float activity;
	float a;
	float alpha;
	float beta;
	float kapp;
	float kapphi;
	float mob;
	float velx1;
	float vely1;
	float velx2;
	float vely2;
	float xf1prev;
	float yf1prev;
	float xf2prev;
	float yf2prev;
			
	// host arrays:
	float* rH;
	float* phi1H;
	float* phi2H;
	float* phi3H;
	float* phisum0H;
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
	float* phi1;
	float* phi2;
	float* phi3;
	float* chempot1;
	float* chempot2;
	float* chempot3;
	float* phisum;
	float* phisum0;
	float2* u;
	float2* F;
	float2* p;
	float2* h;
	tensor2D* stress;
	int* voxelType;
	int* streamIndex;
	int* nList;
	
public:

	class_scsp_active_3phi_D2Q9();
	~class_scsp_active_3phi_D2Q9();
	void allocate();
	void deallocate();
	void allocate_forces();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_lattice_box_periodic();
	void create_lattice_box_shear();
	void stream_index_pull();
	void setU(int,float);
	void setV(int,float);
	void setPx(int,float);
	void setPy(int,float);
	void setR(int,float);
	void setPhi1(int,float);
	void setPhi2(int,float);
	void setPhi3(int,float);
	void setVoxelType(int,int);
	void setPhiSum(float,float,float);
	float getU(int);
	float getV(int);
	float getR(int);
	float getPhi1(int);
	float getPhi2(int);
	float getPhi3(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int);	
	void stream_collide_save_forcing(int,int);
	void stream_collide_save_forcing_varvisc(int,int);
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
	void scsp_active_fluid_update_phi_alternative(int,int);
	void scsp_active_fluid_update_phi_diffusive(int,int);
	void scsp_active_fluid_zero_phisum_3phi(int,int);
	void scsp_active_fluid_sum_phi_3phi(int,int);
	void scsp_active_fluid_enforce_conservation_3phi(int,int);
	void scsp_active_fluid_set_velocity_field(int,int);
	void zero_forces(int,int);	
	void write_output(std::string,int,int,int);
	void write_output_droplet_properties(int);

};

# endif  // CLASS_SCSP_ACTIVE_3PHI_D2Q9_H