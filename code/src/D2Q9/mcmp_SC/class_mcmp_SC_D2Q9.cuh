
# ifndef CLASS_MCMP_SC_D2Q9_H
# define CLASS_MCMP_SC_D2Q9_H

# include "../init/lattice_builders_D2Q9.cuh"
# include "../init/stream_index_builder_D2Q9.cuh"
# include "../init/stream_index_builder_bb_D2Q9.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../iolets/boundary_condition_iolet.cuh"
# include "kernels_mcmp_SC_D2Q9.cuh"
# include <cuda.h>
# include <string>



class class_mcmp_SC_D2Q9 {
	
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
	float* prH;
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
	float* pr;
	float* f1A;
	float* f2A;
	float* f1B;
	float* f2B;
	float* FxA;
	float* FyA;
	float* FxB;
	float* FyB;
	int* x;
	int* y;
	int* nList;
	int* voxelType;
	int* streamIndex;
	iolet2D* iolets;
	
public:

	class_mcmp_SC_D2Q9();
	~class_mcmp_SC_D2Q9();
	void allocate();
	void deallocate();
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
	void setX(int,int);
	void setY(int,int);
	void setRA(int,float);
	void setRB(int,float);
	void setVoxelType(int,int);
	float getU(int);
	float getV(int);
	float getRA(int);
	float getRB(int);
	void swap_populations();	
	void initial_equilibrium(int,int);
	void compute_density(int,int);
	void compute_SC_forces_1(int,int);
	void compute_SC_forces_2(int,int);
	void compute_SC_pressure(int,int);
	void compute_velocity(int,int);
	void collide_stream(int,int);
	void write_output(std::string,int);

};

# endif  // CLASS_MCMP_SC_D2Q9_H