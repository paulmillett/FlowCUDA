
# ifndef CLASS_SCSP_PSM_D3Q19_H
# define CLASS_SCSP_PSM_D3Q19_H

# include "../init/lattice_builders_D3Q19.cuh"
# include "../init/bounding_box_nList_construct_D3Q19.cuh"
# include "../init/stream_index_builder_D3Q19.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_scsp_psm_D3Q19.cuh"
# include "sphere.h"
# include <cuda.h>
# include <string>

class class_scsp_psm_D3Q19 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int nSpheres;
	float nu;
	float dt;
	bool forceFlag;
		
	// host arrays:
	float* uH;
	float* vH;
	float* wH;
	float* rH;
	int* nListH;
	int* streamIndexH;
	int* voxelTypeH;
	sphere* spheresH;
		
	// device arrays:
	float* u;
	float* v;
	float* w;
	float* r;
	float* f1;
	float* f2;
	float* Fx;
	float* Fy;
	float* Fz;
	float* eps;
	int* pID;
	int* streamIndex;
	sphere* spheres;
		
public:

	class_scsp_psm_D3Q19();
	~class_scsp_psm_D3Q19();
	void allocate();
	void deallocate();
	void allocate_forces();
	void memcopy_host_to_device();	
	void memcopy_device_to_host();
	void create_lattice_box();
	void create_lattice_box_periodic();
	void create_lattice_box_shear();
	void create_lattice_box_slit();
	void create_lattice_box_channel();	
	void stream_index_pull();
	void setNu(float);
	void setU(int,float);
	void setV(int,float);
	void setW(int,float);
	void setR(int,float);
	float getU(int);
	float getV(int);
	float getW(int);
	float getR(int);
	void initial_equilibrium(int,int);
	void stream_collide_save(int,int);
	void stream_collide_save_forcing(int,int);	
	void set_boundary_shear_velocity(float,float,int,int);
	void set_boundary_slit_velocity(float,int,int);
	void set_channel_wall_velocity(float,int,int);
	void set_boundary_slit_density(int,int);
	void set_boundary_duct_density(int,int);
	void zero_forces(int,int);
	void add_body_force(float,float,float,int,int);	
	void add_body_force_kolmogorov(float,int,int);	
	void vtk_structured_output_ruvw(std::string,int,int,int,int,int);	

};

# endif  // CLASS_SCSP_PSM_D3Q19_H