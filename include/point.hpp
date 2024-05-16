#ifndef __POINT_HPP__
#define __POINT_HPP__


struct Point2D {
	double x;
	double y;

	Point2D() {};
	Point2D(double x_, double y_) : x(x_), y(y_) {};
};


struct Point3D {
	double x;
	double y;
	double z;

	Point3D() {};
	Point3D(double x_, double y_, double z_) : x(x_) , y(y_) , z(z_)  {};
};


#endif
