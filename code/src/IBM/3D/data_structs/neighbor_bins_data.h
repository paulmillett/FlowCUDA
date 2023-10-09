# ifndef NEIGHBOR_BINS_DATA_H
# define NEIGHBOR_BINS_DATA_H



// --------------------------------------------------------
// struct that defines binned neighbor data:
// --------------------------------------------------------

struct bindata {
	int nBins;
	int nnbins;
	int binMax;
	int3 numBins;
	float sizeBins;
	int* binMembers;
	int* binOccupancy;
	int* binMap;
};


# endif  // NEIGHBOR_BINS_DATA_H