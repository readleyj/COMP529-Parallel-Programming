#pragma once
#include <vector>
#include <array>
#include <functional>
#include <math.h>

namespace MeshReconstruction
{
	struct Vec3
	{
		double x, y, z;

		Vec3 operator+(Vec3 const& o) const
		{
			return { x + o.x, y + o.y, z + o.z };
		}

		Vec3 operator-(Vec3 const& o) const
		{
			return { x - o.x, y - o.y, z - o.z };
		}

		Vec3 operator*(double c) const
		{
			return { c*x, c*y, c*z };
		}

		double Norm() const
		{
			return sqrt(x*x + y*y + z*z);
		}

		Vec3 Normalized() const
		{
			auto n = Norm();
			return { x / n, y / n, z / n };
		}
	};

	struct Rect3
	{
		Vec3 min;
		Vec3 size;
	};

	using Triangle = std::array<int, 3>;

	struct Mesh
	{
		std::vector<Vec3> vertices;
		std::vector<Triangle> triangles;
		std::vector<Vec3> vertexNormals;
	};


	inline double Norm(float p1, float p2)
	{
		return sqrt(p1*p1 + p2*p2);
	};

	using Fun3s = std::function<double(Vec3 const&)>;
	using FunTr = std::function<double(Fun3s const&, Vec3 const&, double)>;
	using Fun3v = std::function<Vec3(Vec3 const&, double)>;
}
