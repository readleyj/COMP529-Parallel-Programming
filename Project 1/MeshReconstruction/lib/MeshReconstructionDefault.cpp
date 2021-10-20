#include "MeshReconstruction.h"
#include "Cube.h"
#include "Triangulation.h"
#include <iostream>

using namespace MeshReconstruction;
using namespace std;

// Adapted from here: http://paulbourke.net/geometry/polygonise/
namespace
{
	Vec3 NumGradDefault(Fun3s const& f, FunTr const& tr, Vec3 const& p, double twist)
	{
		const double Eps = 1e-6;
		Vec3 epsX{ Eps, 0, 0 }, epsY{ 0, Eps, 0 }, epsZ{ 0, 0, Eps };
		double gx = (tr(f, p + epsX, twist) - tr(f, p - epsX, twist)) / 2;
		double gy = (tr(f, p + epsY, twist) - tr(f ,p - epsY, twist)) / 2;
		double gz = (tr(f, p + epsZ, twist) - tr(f, p - epsZ, twist)) / 2;
		return { gx, gy, gz };
	}
}


/// Given a grid cube and an isolevel the triangles (5 max)
/// required to represent the isosurface in the cube are computed.
void TriangulateDefault(
	IntersectInfo const& intersect,
	Fun3v const& grad,
	double twist,
	Mesh& mesh)
{
	// Cube is entirely in/out of the surface. Generate no triangles.
	if (intersect.signConfig == 0 || intersect.signConfig == 255) return;

	const int * tri = signConfigToTriangles[intersect.signConfig];

	for (int i = 0; tri[i] != -1; i += 3)
	{
		Vec3 v0 = intersect.edgeVertIndices[tri[i]];
		Vec3 v1 = intersect.edgeVertIndices[tri[i + 1]];
		Vec3 v2 = intersect.edgeVertIndices[tri[i + 2]];

		Vec3 normal0 = grad(v0, twist).Normalized();
		Vec3 normal1 = grad(v1, twist).Normalized();
		Vec3 normal2 = grad(v2, twist).Normalized();

		mesh.vertices.push_back(v0);
		mesh.vertices.push_back(v1);
		mesh.vertices.push_back(v2);

		int last = static_cast<int>(mesh.vertices.size() - 1);

		mesh.vertexNormals.push_back(normal0);
		mesh.vertexNormals.push_back(normal1);
		mesh.vertexNormals.push_back(normal2);

		mesh.triangles.push_back({ last - 2, last - 1, last });
	}
}


Mesh MeshReconstruction::MarchCubeDefault(
	Fun3s const& sdf,
	FunTr const& transform,
	Rect3 const& domain,
	Vec3 const& cubeSize,
	double twist,
	double isoLevel,
	Fun3v sdfGrad)
{
	// Default value.
	sdfGrad = sdfGrad == nullptr
		? [&sdf, &transform](Vec3 const& p, double twist) { return NumGradDefault(sdf, transform, p, twist); }
		: sdfGrad;

	int NumX = static_cast<int>(ceil(domain.size.x / cubeSize.x));
	int NumY = static_cast<int>(ceil(domain.size.y / cubeSize.y));
	int NumZ = static_cast<int>(ceil(domain.size.z / cubeSize.z));

	double HalfCubeDiag = cubeSize.Norm() / 2.0;
	Vec3 HalfCubeSize = cubeSize * 0.5;

	Mesh mesh;

	// iterate over each cube in the mesh
	for (int ix = 0; ix < NumX; ++ix)
	{
		double x = domain.min.x + ix * cubeSize.x;

		for (int iy = 0; iy < NumY; ++iy)
		{
			double y = domain.min.y + iy * cubeSize.y;

			for (int iz = 0; iz < NumZ; ++iz)
			{
				double z = domain.min.z + iz * cubeSize.z;
				Vec3 min{ x, y, z };

				Cube cube({ min, cubeSize }, sdf, transform, twist);
				//map the vertices under the isosurface to intersecting edges
				IntersectInfo intersect = cube.Intersect(isoLevel);
				//now create and store the triangle data
				TriangulateDefault(intersect, sdfGrad, twist, mesh);
			}
		}
	}

	return mesh;
}
