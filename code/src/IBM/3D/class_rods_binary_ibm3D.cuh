
# ifndef CLASS_RODS_BINARY_IBM3D_H
# define CLASS_RODS_BINARY_IBM3D_H

# include "../../IO/read_ibm_information.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../Utils/helper_math.h"
# include "../../D3Q19/scsp/class_scsp_D3Q19.cuh"
# include "class_rods_ibm3D.cuh"
# include "kernels_rods_ibm3D.cuh"
# include "data_structs/cell_data.h"
# include <cuda.h>
# include <string>


// ---------------------------------------------------------------------------
// Note: this class inherits from "class_capsule_ibm3D" and should
//       be used for binary suspensions (i.e. two different types of capsules)
// ---------------------------------------------------------------------------


class class_rods_binary_ibm3D : public class_rods_ibm3D {
	
	public:  // treat like a struct
	
	// data:
	int nRods1;
	int nRods2;
	int nBeadsPerRod1;
	int nBeadsPerRod2;	
		
	// methods:
	class_rods_binary_ibm3D();
	~class_rods_binary_ibm3D();
	void create_first_rod();
	void duplicate_rods();
	void set_aspect_ratios(float,float);
	void set_mobility_coefficients(float,float,float,float,float);
};

# endif  // CLASS_RODS_BINARY_IBM3D_H