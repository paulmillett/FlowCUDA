
# ifndef CLASS_MCMP_SC_BB_D2Q9_H
# define CLASS_MCMP_SC_BB_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include "../../Lattice/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "../init/stream_index_builder_bb_D2Q9.cuh"
# include "mcmp_initial_equilibrium_bb_D2Q9.cuh"
# include "mcmp_initial_particles_on_lattice_D2Q9.cuh"
# include "mcmp_compute_density_bb_D2Q9.cuh"
# include "mcmp_update_particles_on_lattice_D2Q9.cuh"
# include "mcmp_compute_SC_forces_bb_D2Q9.cuh"
# include "mcmp_compute_velocity_bb_D2Q9.cuh"
# include "mcmp_collide_stream_bb_D2Q9.cuh"
# include "mcmp_bounce_back_D2Q9.cuh"
# include "mcmp_bounce_back_moving_D2Q9.cuh"
# include "../../IO/write_vtk_output.cuh"
# include <cuda.h>
# include <string>

class class_mcmp_SC_bb_D2Q9 {
	
private:

	// scalars: 
	int Q;
	int nVoxels;	
	int Nx,Ny,Nz;
	int numIolets;
	float nu;
	float gAB;
	float gAS;
	float gBS;
			
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
	
	// device arrays:
	float* u;
	float* v;
	float* rA;
	float* rB;
	float* f1A;
	float* f2A;
	float* f1B;
	float* f2B;
	float* FxA;
	float* FyA;
	float* FxB;
	float* FyB;
	int* s;
	int* x;
	int* y;
	int* nList;
	int* voxelType;
	int* streamIndex;
	iolet2D* iolets;
	
public:

	class_mcmp_SC_bb_D2Q9();
	~class_mcmp_SC_bb_D2Q9();
	void allocate();
	void deallocate();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void create_lattice_box();
	void create_lattice_box_periodic();
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
	float getU(int);
	float getV(int);
	int   getS(int);
	float getRA(int);
	float getRB(int);
	void initial_equilibrium_bb(int,int);
	void initial_equilibrium_psm(int,int);
	void initial_particles_on_lattice(float*,float*,float*,int*,int,int,int);
	void compute_density_bb(int,int);
	void update_particles_on_lattice(float*,float*,float*,float*,float*,int*,int,int,int);
	void compute_SC_forces_bb(int,int);
	void compute_velocity_bb(int,int);
	void collide_stream_bb(int,int);
	void bounce_back(int,int);
	void bounce_back_moving(int,int);
	void swap_populations();	
	
	void write_output(std::string,int);

};

# endif  // CLASS_MCMP_SC_BB_D2Q9_H