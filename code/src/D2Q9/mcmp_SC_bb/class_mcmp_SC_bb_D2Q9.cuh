
# ifndef CLASS_MCMP_SC_BB_D2Q9_H
# define CLASS_MCMP_SC_BB_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include "../init/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "../init/stream_index_builder_bb_D2Q9.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_mcmp_SC_bb_D2Q9.cuh"
# include <cuda.h>
# include <string>


class class_mcmp_SC_bb_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	int nParts;
	float nu;
	float gAB;
	float gAS;
	float gBS;
	float omega;
	float rAsum,rAsum0;
	float rBsum,rBsum0;
			
	// host arrays:
	float* uH;
	float* vH;
	float* rAH;
	float* rBH;
	int* sH;
	int* xH;
	int* yH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	iolet2D* ioletsH;
	particle2D_bb* ptH;
	
	// device arrays:
	float* u;
	float* v;
	float* rA;
	float* rB;
	float* rAvirt;
	float* rBvirt;
	float* f1A;
	float* f2A;
	float* f1B;
	float* f2B;
	float* FxA;
	float* FyA;
	float* FxB;
	float* FyB;
	int* s;
	int* sprev;
	int* x;
	int* y;
	int* nList;
	int* voxelType;
	int* streamIndex;
	int* pIDgrid;
	iolet2D* iolets;
	particle2D_bb* pt;
	
public:

	class_mcmp_SC_bb_D2Q9();
	~class_mcmp_SC_bb_D2Q9();
	void allocate();
	void deallocate();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void memcopy_device_to_host_particles();
	void create_lattice_box();
	void create_lattice_box_periodic();
	void create_lattice_box_shear();
	void create_lattice_file();
	void stream_index_push();
	void stream_index_pull();
	void stream_index_push_bb();
	void read_iolet_info(int,const char*);
	void setU(int,float);
	void setV(int,float);
	void setS(int,int);
	void setX(int,int);
	void setY(int,int);
	void setRA(int,float);
	void setRB(int,float);
	void setVoxelType(int,int);
	void setPrx(int,float);
	void setPry(int,float);
	void setPvx(int,float);
	void setPvy(int,float);
	void setPrad(int,float);
	void setPmass(int,float);
	float getU(int);
	float getV(int);
	int   getS(int);
	float getRA(int);
	float getRB(int);
	float getPrx(int);
	float getPry(int);
	float getPfx(int);
	float getPfy(int);
	float getPrad(int);
	float getPmass(int);
	void calculate_initial_density_sums();
	void zero_particle_forces_bb(int,int);
	void move_particles_bb(int,int);
	void fix_particle_velocity_bb(float,int,int);
	void particle_particle_forces_bb(float,float,int,int);
	void initial_equilibrium_bb(int,int);
	void map_particles_on_lattice_bb(int,int);
	void cover_uncover_bb(int,int);
	void compute_density_bb(int,int);
	void correct_density_totals_bb(int,int);
	void compute_virtual_density_bb(int,int);
	void update_particles_on_lattice_bb(int,int);
	void compute_SC_forces_bb(int,int);
	void compute_SC_forces_bb_2(int,int);
	void compute_velocity_bb(int,int);
	void set_boundary_velocity_bb(float,float,int,int);
	void set_boundary_shear_velocity_bb(float,float,int,int);
	void collide_stream_bb(int,int);
	void bounce_back(int,int);
	void bounce_back_moving(int,int);
	void swap_populations();
	void sum_fluid_densities_bb(int,int);		
	void write_output(std::string,int);
	void write_density_sums(int);
	
};

# endif  // CLASS_MCMP_SC_BB_D2Q9_H