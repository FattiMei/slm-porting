#ifndef __TYPES_H__
#define __TYPES_H__


#include <vector>
#include <concepts>
#include <type_traits>


template <typename T>
concept SpotLike = requires (T a) {
	{ a.x } -> std::convertible_to<double>;
	{ a.y } -> std::convertible_to<double>;
	{ a.z } -> std::convertible_to<double>;
};


template <typename T>
concept SpotContainer = requires (T container) {
	{ container[std::declval<int>()] };
	{ container.size() } -> std::integral;

	requires SpotLike<std::remove_cvref_t<decltype(container[0])>>;
};


// Spot is a naturally AoS (array of structs) layout for spots
// to properly compare with the python implementations we need
// a SoA container
struct Spot {
	double x;
	double y;
	double z;

	bool operator==(const Spot& rhs) const {
		return x == rhs.x and
		       y == rhs.y and
		       z == rhs.z;
	}
};


struct SpotAligned {
	SpotAligned(double x_, double y_, double z_) : x(x_),
	                                               y(y_),
	                                               z(z_),
	                                               unused(0.0) {}

	bool operator==(const Spot& rhs) const {
		const Spot view(x, y, z);
		return view == rhs;
	}

	double x;
	double y;
	double z;
	double unused;
};


template <SpotLike S>
class SpotSoaContainer {
	public:
		SpotSoaContainer(const SpotContainer auto& v) {
			m_x.reserve(v.size());
			m_y.reserve(v.size());
			m_z.reserve(v.size());

			for (const auto& spot : v) {
				m_x.push_back(spot.x);
				m_y.push_back(spot.y);
				m_z.push_back(spot.z);
			}
		}

		SpotSoaContainer(std::vector<double>&& x,
		                 std::vector<double>&& y,
		                 std::vector<double>&& z) : m_x(x),
		                                            m_y(y),
		                                            m_z(z) {}

		std::size_t size() const {
			return m_x.size();
		}

		const auto operator[](std::integral auto i) const {
			return S(m_x[i], m_y[i], m_z[i]);
		}

	private:
		std::vector<double> m_x;
		std::vector<double> m_y;
		std::vector<double> m_z;
};


#endif  // __TYPES_H__
