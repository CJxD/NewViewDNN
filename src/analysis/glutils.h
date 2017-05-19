#pragma once

#ifdef __APPLE__
#include <OpenGL/glew.h>
#include <OpenGL/gl.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <vector>

typedef float scalar_t;

struct DrawObject
{
	struct Material {
		scalar_t ambient[3] {0,0,0};
		scalar_t diffuse[3] {0,0,0};
		scalar_t specular[3] {0,0,0};

		scalar_t shininess = 0;
		scalar_t roughness = 0;

		GLuint ambientTex = 0;
		GLuint diffuseTex = 0;
		GLuint specularTex = 0;
		GLuint bumpTex = 0;
	};

	static constexpr const scalar_t* const vertex_ptr = 0;
	static constexpr const size_t vertex_size = 3;
	static constexpr const scalar_t* const normal_ptr = vertex_ptr + vertex_size;
	static constexpr const size_t normal_size = 3;
	static constexpr const scalar_t* const color_ptr = normal_ptr + normal_size;
	static constexpr const size_t color_size = 3;
	static constexpr const scalar_t* const uv_ptr = color_ptr + color_size;
	static constexpr const size_t uv_size = 2;
	static constexpr const size_t stride = (vertex_size + normal_size
			+ color_size + uv_size) * sizeof(scalar_t);

	GLuint vb; // Vertex buffer
	scalar_t bmin[3], bmax[3]; // Bounding box
	int numTriangles;
	Material material;
};

// Function to check OpenGL error status
#define glTry(func) func; _check(__FILE__, __LINE__, #func)
void _check(const char* file, int line, const char* where);

// Function to check OpenGL shader error status
#define checkShader(shader) _checkShader(shader, __FILE__, __LINE__)
void _checkShader(GLuint shader, const char* file, int line);

// Function to check OpenGL program error status
#define checkProgram(program) _checkProgram(program, __FILE__, __LINE__)
void _checkProgram(GLuint program, const char* file, int line);

// Function to calculate the surface normal of a triangular face
void calcNormal(scalar_t N[3], scalar_t v0[3], scalar_t v1[3], scalar_t v2[3]);

// A simple function that will read a file into an allocated char pointer buffer
char* file2buf(const char* filename);

// Loads an image into a new texture buffer
GLuint loadTexture(const char* filename);

// Loads a Wavefront .obj object into a new vertex buffer
std::vector<DrawObject> loadWavefront(const char* filename);
