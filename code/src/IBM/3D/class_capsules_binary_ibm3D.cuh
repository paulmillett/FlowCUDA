
# ifndef CLASS_CAPSULES_BINARY_IBM3D_H
# define CLASS_CAPSULES_BINARY_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "class_capsules_ibm3D.cuh"
# include "kernels_ibm3D.cuh"
# include "kernels_capsules_ibm3D.cuh"
# include "kernels_nonbonded_ibm3D.cuh"
# include "data_structs/cell_data.h"
# include <cuda.h>
# include <string>


// ---------------------------------------------------------------------------
// Note: this class inherits from "class_capsule_ibm3D" and should
//       be used for binary suspensions (i.e. two different types of capsules)
// ---------------------------------------------------------------------------


class class_capsules_binary_ibm3D : public class_capsules_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nCells1;
	int nCells2;
	int nNodesPerCell1;
	int nFacesPerCell1;	
	int nEdgesPerCell1;
	int nNodesPerCell2;
	int nFacesPerCell2;	
	int nEdgesPerCell2;
	float a1;
	float a2;
	
	// methods:
	class_capsules_binary_ibm3D();
	~class_capsules_binary_ibm3D();
	void read_ibm_information(std::string,std::string);
	void read_ibm_file_long(std::string,int,int,int,int,int,int,int);
	void duplicate_cells();	
	void set_cells_radii_binary();
	void set_cells_types_binary();
	void randomize_platelets_and_rbcs(float,float);
	void randomize_probe_and_rbcs(float,float);
	void stepIBM_no_fluid_rbcs_platelets(int,bool,int,int);
	void update_node_positions_verlet_1_cellType2_stationary(int,int);
	
};

# endif  // CLASS_CAPSULES_BINARY_IBM3D_H