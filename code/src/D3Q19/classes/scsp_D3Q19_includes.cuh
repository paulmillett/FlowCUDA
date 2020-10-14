# ifndef SCSP_D3Q19_INCLUDES_H
# define SCSP_D3Q19_INCLUDES_H


# include "../../Lattice/lattice_builders_D3Q19.cuh"
# include "../../Lattice/bounding_box_nList_construct_D3Q19.cuh"
# include "../init/stream_index_builder_D3Q19.cuh"
# include "../scsp/scsp_initial_equilibrium_D3Q19.cuh"
# include "../scsp/scsp_stream_collide_save_D3Q19.cuh"
# include "../scsp/scsp_stream_collide_save_IBforcing_D3Q19.cuh"
# include "../scsp/scsp_stream_collide_save_forcing_D3Q19.cuh"
# include "../scsp/scsp_zero_forces_D3Q19.cuh"
# include "../inout/inside_hemisphere_D3Q19.cuh"
# include "../../IO/write_vtk_output.cuh"
# include "../../IO/read_lattice_geometry.cuh"
# include "../../IBM/3D/extrapolate_velocity_IBM3D.cuh"


# endif  // SCSP_D3Q19_INCLUDES_H