#include "IO.h"
using namespace std;

void MeshReconstruction::WriteObjFile(Mesh const& mesh, string const& fileName)
{
	// FILE faster than streams.

	FILE* file = fopen(fileName.c_str(), "w");
	if (!file)
	{
		throw runtime_error("Could not write obj file.");
	}

	// write stats
	fprintf(file, "# %d vertices, %d triangles\n\n",
		static_cast<int>(mesh.vertices.size()),
		static_cast<int>(mesh.triangles.size()));

	// vertices
	for (int vi = 0; vi < mesh.vertices.size(); ++vi)
	{
		Vec3 const& v = mesh.vertices.at(vi);
		fprintf(file, "v %f %f %f\n", v.x, v.y, v.z);
	}

	// vertex normals
	fprintf(file, "\n");
	for (int ni = 0; ni < mesh.vertices.size(); ++ni)
	{
		Vec3 const& vn = mesh.vertexNormals.at(ni);
		fprintf(file, "vn %f %f %f\n", vn.x, vn.y, vn.z);
	}

	// triangles (1-based)
	fprintf(file, "\n");
	for (int ti = 0; ti < mesh.triangles.size(); ++ti)
	{
		Triangle const& t = mesh.triangles.at(ti);
		fprintf(file, "f %d//%d %d//%d %d//%d\n",
			t[0] + 1, t[0] + 1,
			t[1] + 1, t[1] + 1,
			t[2] + 1, t[2] + 1);
	}

	fclose(file);
}
