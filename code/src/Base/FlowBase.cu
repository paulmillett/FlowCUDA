
# include "FlowBase.cuh"
# include <iostream>

// -------------------------------------------------------------------------
// List of header files that need to be included...
// -------------------------------------------------------------------------

# include "../Apps/mcmp_2D_basic.cuh"
# include "../Apps/mcmp_2D_capbridge.cuh"
# include "../Apps/mcmp_2D_dropsurface.cuh"
# include "../Apps/mcmp_2D_dropsurface_bb.cuh"
# include "../Apps/mcmp_2D_particle_bb.cuh"
# include "../Apps/mcmp_2D_particle_dip.cuh"
# include "../Apps/mcmp_2D_capbridge_psm.cuh"
# include "../Apps/mcmp_2D_capbridge_dip.cuh"
# include "../Apps/mcmp_2D_capbridge_bb.cuh"
# include "../Apps/mcmp_2D_capbridge_move_bb.cuh"
# include "../Apps/mcmp_2D_capbridge_shear_bb.cuh"
# include "../Apps/mcmp_2D_capbridge_extensional_bb.cuh"
# include "../Apps/mcmp_2D_doublet_shear_bb.cuh"
# include "../Apps/mcmp_2D_ellipse_shear_bb.cuh"
# include "../Apps/mcmp_2D_drag_dip.cuh"
# include "../Apps/mcmp_2D_drag_bb.cuh"
# include "../Apps/scsp_2D_obstacle.cuh"
# include "../Apps/scsp_2D_expand.cuh"
# include "../Apps/scsp_2D_expand_2.cuh"
# include "../Apps/scsp_3D_iolets.cuh"
# include "../Apps/scsp_3D_hemisphere_2.cuh"
# include "../Apps/scsp_3D_hemisphere_3.cuh"
# include "../Apps/scsp_3D_bulge.cuh"
# include "../Apps/scsp_3D_capsule.cuh"
# include "../Apps/scsp_3D_capsule_skalak.cuh"
# include "../Apps/scsp_3D_capsule_skalak_verlet.cuh"
# include "../Apps/scsp_3D_capsule_sedimentation.cuh"
# include "../Apps/scsp_3D_capsule_visc_contrast.cuh"
# include "../Apps/scsp_3D_two_capsules.cuh"
# include "../Apps/scsp_3D_capsules_lots.cuh"
# include "../Apps/scsp_3D_capsules_susp_shear.cuh"
# include "../Apps/scsp_3D_capsules_constriction.cuh"
# include "../Apps/scsp_3D_capsules_channel.cuh"
# include "../Apps/scsp_3D_capsules_slit.cuh"
# include "../Apps/scsp_3D_capsules_slit_trains.cuh"
# include "../Apps/scsp_3D_capsules_slit_membrane.cuh"
# include "../Apps/scsp_3D_duct_channel.cuh"
# include "../Apps/scsp_3D_slit_channel.cuh"
# include "../Apps/scsp_3D_rbcs_susp_shear.cuh"
# include "../Apps/scsp_3D_swell_ibm.cuh"
# include "../Apps/scsp_3D_leftvent_ibm.cuh"
# include "../Apps/mcmp_3D_capbridge_dip.cuh"
# include "../Apps/mcmp_3D_capbridge_bb.cuh"
# include "../Apps/mcmp_3D_capbridge_shear_bb.cuh"



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
	if (specifier == "mcmp_2D_particle_bb") return new mcmp_2D_particle_bb();
	if (specifier == "mcmp_2D_particle_dip") return new mcmp_2D_particle_dip();
	if (specifier == "mcmp_2D_capbridge_psm") return new mcmp_2D_capbridge_psm();
	if (specifier == "mcmp_2D_capbridge_dip") return new mcmp_2D_capbridge_dip();
	if (specifier == "mcmp_2D_capbridge_bb") return new mcmp_2D_capbridge_bb();
	if (specifier == "mcmp_2D_capbridge_move_bb") return new mcmp_2D_capbridge_move_bb();
	if (specifier == "mcmp_2D_capbridge_shear_bb") return new mcmp_2D_capbridge_shear_bb();
	if (specifier == "mcmp_2D_capbridge_extensional_bb") return new mcmp_2D_capbridge_extensional_bb();
	if (specifier == "mcmp_2D_doublet_shear_bb") return new mcmp_2D_doublet_shear_bb();
	if (specifier == "mcmp_2D_ellipse_shear_bb") return new mcmp_2D_ellipse_shear_bb();
	if (specifier == "mcmp_2D_drag_dip") return new mcmp_2D_drag_dip();
	if (specifier == "mcmp_2D_drag_bb") return new mcmp_2D_drag_bb();
	if (specifier == "scsp_2D_obstacle") return new scsp_2D_obstacle();
	if (specifier == "scsp_2D_expand") return new scsp_2D_expand();
	if (specifier == "scsp_2D_expand_2") return new scsp_2D_expand_2();
	if (specifier == "scsp_3D_iolets") return new scsp_3D_iolets();
	if (specifier == "scsp_3D_hemisphere_2") return new scsp_3D_hemisphere_2();
	if (specifier == "scsp_3D_hemisphere_3") return new scsp_3D_hemisphere_3();
	if (specifier == "scsp_3D_bulge") return new scsp_3D_bulge();
	if (specifier == "scsp_3D_capsule") return new scsp_3D_capsule();
	if (specifier == "scsp_3D_capsule_skalak") return new scsp_3D_capsule_skalak();
	if (specifier == "scsp_3D_capsule_skalak_verlet") return new scsp_3D_capsule_skalak_verlet();
	if (specifier == "scsp_3D_capsule_sedimentation") return new scsp_3D_capsule_sedimentation();
	if (specifier == "scsp_3D_capsule_visc_contrast") return new scsp_3D_capsule_visc_contrast();
	if (specifier == "scsp_3D_two_capsules") return new scsp_3D_two_capsules();
	if (specifier == "scsp_3D_capsules_lots") return new scsp_3D_capsules_lots();
	if (specifier == "scsp_3D_capsules_susp_shear") return new scsp_3D_capsules_susp_shear();
	if (specifier == "scsp_3D_capsules_constriction") return new scsp_3D_capsules_constriction();
	if (specifier == "scsp_3D_capsules_channel") return new scsp_3D_capsules_channel();
	if (specifier == "scsp_3D_capsules_slit") return new scsp_3D_capsules_slit();
	if (specifier == "scsp_3D_capsules_slit_trains") return new scsp_3D_capsules_slit_trains();
	if (specifier == "scsp_3D_capsules_slit_membrane") return new scsp_3D_capsules_slit_membrane();
	if (specifier == "scsp_3D_duct_channel") return new scsp_3D_duct_channel();
	if (specifier == "scsp_3D_slit_channel") return new scsp_3D_slit_channel();
	if (specifier == "scsp_3D_rbcs_susp_shear") return new scsp_3D_rbcs_susp_shear();
	if (specifier == "scsp_3D_swell_ibm") return new scsp_3D_swell_ibm();
	if (specifier == "scsp_3D_leftvent_ibm") return new scsp_3D_leftvent_ibm();
	if (specifier == "mcmp_3D_capbridge_dip") return new mcmp_3D_capbridge_dip();
	if (specifier == "mcmp_3D_capbridge_bb") return new mcmp_3D_capbridge_bb();
	if (specifier == "mcmp_3D_capbridge_shear_bb") return new mcmp_3D_capbridge_shear_bb();
   
	// -----------------------------------
	// if input file doesn't have a 
	// correct type return a nullptr
	// -----------------------------------
   
	return NULL;

}
