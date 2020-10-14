
# ifndef STRUCT_IBM3D_H
# define STRUCT_IBM3D_H

# include <cuda.h>
# include <string>

struct struct_ibm3D {
	
	// data:
	int nNodes;
	int nFaces; 
	float k_stiff;	
	bool facesFlag;
		
	// host arrays:
	float* xH;
	float* yH;
	float* zH;
	float* xH_start;
	float* yH_start;
	float* zH_start;
	float* xH_end;
	float* yH_end;
	float* zH_end;
	int* faceV1;
	int* faceV2;
	int* faceV3;
		
	// device arrays:
	float* x;
	float* y;
	float* z;
	float* x_start;
	float* y_start;
	float* z_start;
	float* x_end;
	float* y_end;
	float* z_end;	
	float* vx;
	float* vy;
	float* vz;
	float* fx;
	float* fy;
	float* fz;	
	
	// methods:
	struct_ibm3D();
	~struct_ibm3D();
	void allocate();
	void deallocate();	
	void allocate_faces();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void read_ibm_start_positions(std::string);
	void read_ibm_end_positions(std::string);
	void initialize_positions_to_start();
	void shift_start_positions(float,float,float);
	void shift_end_positions(float,float,float);
	void write_output(std::string,int);
	void update_node_positions(int,int,int,int);	

};

# endif  // STRUCT_IBM3D_H