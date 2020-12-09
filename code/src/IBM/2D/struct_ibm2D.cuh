
# ifndef STRUCT_IBM2D_H
# define STRUCT_IBM2D_H

# include <cuda.h>
# include <string>

struct struct_ibm2D {
	
	// data:
	int nNodes;
	int nFaces; 
	float kstiff;	
	bool facesFlag;
		
	// host arrays:
	float* xH;
	float* yH;
	float* xH_start;
	float* yH_start;
	float* xH_end;
	float* yH_end;
	int* faceV1;
	int* faceV2;
	int* faceV3;
		
	// device arrays:
	float* x;
	float* y;
	float* x_start;
	float* y_start;
	float* x_end;
	float* y_end;
	float* x_ref;
	float* y_ref;
	float* vx;
	float* vy;
	float* fx;
	float* fy;
	
	// methods:
	struct_ibm2D();
	~struct_ibm2D();
	void allocate();
	void deallocate();	
	void allocate_faces();
	void memcopy_host_to_device();
	void memcopy_device_to_host();
	void setXStart(int,float);
	void setYStart(int,float);
	void setXEnd(int,float);
	void setYEnd(int,float);
	void set_positions_to_start_positions();
	void read_ibm_start_positions(std::string);
	void read_ibm_end_positions(std::string);
	void initialize_positions_to_start();
	void shift_start_positions(float,float);
	void shift_end_positions(float,float);
	void write_output(std::string,int);
	void set_reference_node_positions(int,int);
	void compute_node_forces(int,int);
	void update_node_ref_position(int,int);
	void update_node_ref_position(int,int,int,int);
	void update_node_positions(int,int);
	void update_node_positions(int,int,int,int);	

};

# endif  // STRUCT_IBM2D_H