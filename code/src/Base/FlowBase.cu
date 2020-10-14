
# include "FlowBase.cuh"
# include <iostream>

// -------------------------------------------------------------------------
// List of header files that need to be included...
// -------------------------------------------------------------------------

# include "../Apps/mcmp_2D_basic.cuh"
# include "../Apps/mcmp_2D_capbridge.cuh"
# include "../Apps/mcmp_2D_dropsurface.cuh"
# include "../Apps/mcmp_2D_dropsurface_bb.cuh"
# include "../Apps/scsp_2D_obstacle.cuh"
# include "../Apps/scsp_2D_expand.cuh"
# include "../Apps/scsp_2D_expand_2.cuh"
# include "../Apps/scsp_3D_iolets.cuh"
# include "../Apps/scsp_3D_iolets_2.cuh"
# include "../Apps/scsp_3D_hemisphere.cuh"
# include "../Apps/scsp_3D_hemisphere_2.cuh"
# include "../Apps/scsp_3D_hemisphere_3.cuh"
# include "../Apps/scsp_3D_obstacle.cuh"
# include "../Apps/scsp_3D_bulge.cuh"
# include "../Apps/scsp_3D_swell_ibm.cuh"
# include "../Apps/scsp_3D_leftvent_ibm.cuh"



// -------------------------------------------------------------------------
// Factory method: this function returns an object determined
// by the string 'specifier':
// {Note: all of the returnable objects inherent from 'FlowBase'}
// -------------------------------------------------------------------------

FlowBase* FlowBase::FlowObjectFactory(string specifier)
{

	// -----------------------------------
	// return the requested object:
	// -----------------------------------

	cout << specifier << endl; 

	if (specifier == "mcmp_2D_basic") return new mcmp_2D_basic();
	if (specifier == "mcmp_2D_capbridge") return new mcmp_2D_capbridge();
	if (specifier == "mcmp_2D_dropsurface") return new mcmp_2D_dropsurface();
	if (specifier == "mcmp_2D_dropsurface_bb") return new mcmp_2D_dropsurface_bb();
	if (specifier == "scsp_2D_obstacle") return new scsp_2D_obstacle();
	if (specifier == "scsp_2D_expand") return new scsp_2D_expand();
	if (specifier == "scsp_2D_expand_2") return new scsp_2D_expand_2();
	if (specifier == "scsp_3D_iolets") return new scsp_3D_iolets();
	if (specifier == "scsp_3D_iolets_2") return new scsp_3D_iolets_2();
	if (specifier == "scsp_3D_hemisphere") return new scsp_3D_hemisphere();
	if (specifier == "scsp_3D_hemisphere_2") return new scsp_3D_hemisphere_2();
	if (specifier == "scsp_3D_hemisphere_3") return new scsp_3D_hemisphere_3();
	if (specifier == "scsp_3D_obstacle") return new scsp_3D_obstacle();
	if (specifier == "scsp_3D_bulge") return new scsp_3D_bulge();
	if (specifier == "scsp_3D_swell_ibm") return new scsp_3D_swell_ibm();
	if (specifier == "scsp_3D_leftvent_ibm") return new scsp_3D_leftvent_ibm();
   
	// -----------------------------------
	// if input file doesn't have a 
	// correct type return a nullptr
	// -----------------------------------
   
	return NULL;

}
