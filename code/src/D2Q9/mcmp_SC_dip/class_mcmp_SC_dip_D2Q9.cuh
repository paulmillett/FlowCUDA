
# ifndef CLASS_MCMP_SC_DIP_D2Q9_H
# define CLASS_MCMP_SC_DIP_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include "../init/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "../init/stream_index_builder_bb_D2Q9.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "kernels_mcmp_SC_dip_D2Q9.cuh"
# include <cuda.h>
# include <string>

class class_mcmp_SC_dip_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nParts;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;	
	float nu;
	float gAB;
	float gAS;	
	float gBS;
	float omega;
	float3 box;
		
	// host arrays:
	float* uH;
	float* vH;
	float* rAH;
	float* rBH;	
	float* pfxH;
	float* pfyH;
	int* xH;
	int* yH;
	int* nListH;
	int* voxelTypeH;
	int* streamIndexH;
	iolet2D* ioletsH;
	particle2D_dip* ptH;
	
	// device arrays:
	float* u;
	float* v;	
	float* rA;
	float* rB;
	float* rS;
	float* f1A;
	float* f2A;
	float* f1B;
	float* f2B;	
	float* FxA;
	float* FxB;
	float* FyA;
	float* FyB;
	float* pfx;
	float* pfy;
	int* x;
	int* y;	
	int* nList;
	int* voxelType;
	int* pIDgrid;
	int* streamIndex;
	iolet2D* iolets;
	particle2D_dip* pt;
	
public:

	class_mcmp_SC_dip_D2Q9();
	~class_mcmp_SC_dip_D2Q9();
	void allocate();
	void deallocate();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void memcopy_device_to_host_particles();
	void create_lattice_box();
	void create_lattice_box_periodic();
	void create_lattice_file();
	void stream_index_push();
	void stream_index_pull();
	void read_iolet_info(int,const char*);
	void setU(int,float);
	void setV(int,float);
	void setX(int,int);
	void setY(int,int);
	void setRA(int,float);
	void setRB(int,float);
	void setVoxelType(int,int);
	void setPrx(int,float);
	void setPry(int,float);
	void setPvx(int,float);
	void setPvy(int,float);
	void setPrInner(int,float);
	void setPrOuter(int,float);
	float getU(int);
	float getV(int);
	float getRA(int);
	float getRB(int);
	float getPrx(int);
	float getPry(int);
	float getPfx(int);
	float getPfy(int);
	float getPrInner(int);
	float getPrOuter(int);
	void initial_equilibrium_dip(int,int);
	void compute_density_dip(int,int);
	void map_particles_to_lattice_dip(int,int);
	void compute_SC_forces_dip(int,int);
	void compute_velocity_dip(int,int);
	void compute_velocity_dip_2(int,int);
	void set_boundary_velocity_dip(float,float,int,int);
	void collide_stream_dip(int,int);
	void zero_particle_forces_dip(int,int);
	void move_particles_dip(int,int);
	void fix_particle_velocity_dip(float,int,int);
	void swap_populations();	
	
	void write_output(std::string,int);

};

# endif  // CLASS_MCMP_SC_DIP_D2Q9_H