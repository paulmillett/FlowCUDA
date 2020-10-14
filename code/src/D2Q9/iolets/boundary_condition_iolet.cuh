# ifndef BOUNDARY_CONDITION_IOLET2D_H
# define BOUNDARY_CONDITION_IOLET2D_H

// --------------------------------------------------------
// This struct contains information needed for each
// in inlet/outlet boundary:
// --------------------------------------------------------

struct iolet2D {
	int type;
	float uBC;
	float vBC;
	float rBC;
	float pBC;
};

# endif  // BOUNDARY_CONDITION_IOLET2D_H