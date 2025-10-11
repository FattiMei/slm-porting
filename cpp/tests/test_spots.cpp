#include <vector>
#include <cassert>
#include <iostream>
#include "types.h"


int main() {
	std::vector<Spot> spots{
		{0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0},
		{1.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0}
	};

	SpotSoaContainer<Spot> soa_spot(spots);
	SpotSoaContainer<SpotAligned> soa_spot_aligned(spots);

	for (int i = 0; i < spots.size(); ++i) {
		assert(spots[i] == soa_spot[i]);
		assert(spots[i] == soa_spot_aligned[i]);
	}

	return 0;
}
