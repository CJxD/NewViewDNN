#include "glutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <map>
#include <limits>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void _check(const char* file, int line, const char* where)
{
	const char * what;
	int err = glGetError(); //0 means no error
	if (!err)
		return;
	if (err == GL_INVALID_ENUM)
		what = "GL_INVALID_ENUM";
	else if (err == GL_INVALID_VALUE)
		what = "GL_INVALID_VALUE";
	else if (err == GL_INVALID_OPERATION)
		what = "GL_INVALID_OPERATION";
	else if (err == GL_INVALID_FRAMEBUFFER_OPERATION)
		what = "GL_INVALID_FRAMEBUFFER_OPERATION";
	else if (err == GL_OUT_OF_MEMORY)
		what = "GL_OUT_OF_MEMORY";
	else
		what = "Unknown Error";
	fprintf(stderr, "Error (%d) %s in %s line %d\n%s\n", err, what, file, line,
			where);
	exit(1);
}

void _checkShader(GLuint shader, const char* file, int line)
{
	GLint isCompiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> err(maxLength);
		glGetShaderInfoLog(shader, maxLength, &maxLength, &err[0]);

		fprintf(stderr, "Shader compile error in %s line %d:\n%s\n", file, line,
				&err[0]);
	}
}

void _checkProgram(GLuint program, const char* file, int line)
{
	GLint isCompiled = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &isCompiled);

	if (isCompiled == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

		std::vector<GLchar> err(maxLength);
		glGetProgramInfoLog(program, maxLength, &maxLength, &err[0]);

		fprintf(stderr, "Program link error in %s line %d:\n%s\n", file, line,
				&err[0]);
		exit(1);
	}
}

char* file2buf(const char* filename)
{
	FILE *fptr;
	long length;
	char *buf;
	fprintf(stderr, "Loading %s\n", filename);
	fptr = fopen(filename, "rb"); /* Open file for reading */
	if (!fptr)
	{ /* Return NULL on failure */
		fprintf(stderr, "failed to open %s\n", filename);
		return NULL;
	}
	fseek(fptr, 0, SEEK_END); /* Seek to the end of the file */
	length = ftell(fptr); /* Find out how many bytes into the file we are */
	buf = (char*) malloc(length + 1); /* Allocate a buffer for the entire length of the file and a null terminator */
	fseek(fptr, 0, SEEK_SET); /* Go back to the beginning of the file */
	fread(buf, length, 1, fptr); /* Read the contents of the file in to the buffer */
	fclose(fptr); /* Close the file */
	buf[length] = 0; /* Null terminator */
	return buf; /* Return the buffer */
}

bool fileExists(const char* filename)
{
	bool ret;

	FILE *fp = fopen(filename, "rb");
	if (fp)
	{
		ret = true;
		fclose(fp);
	}
	else
		ret = false;

	return ret;
}

GLuint loadTexture(const char* filename)
{
	int width, height;
	int comp;
	GLuint id;

	fprintf(stderr, "Loading %s\n", filename);
	stbi_set_flip_vertically_on_load(true);
	unsigned char* image = stbi_load(filename, &width, &height, &comp,
			STBI_default);
	if (!image)
	{
		fprintf(stderr, "Unable to load texture: %s\n", filename);
		exit(1);
	}

	glTry(glGenTextures(1, &id));
	glTry(glBindTexture(GL_TEXTURE_2D, id));

	if (comp == 3)
	{
		glTry(
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image));
	}
	else if (comp == 4)
	{
		glTry(
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image));
	}

	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));

	glTry(glBindTexture(GL_TEXTURE_2D, 0));

	stbi_image_free(image);

	return id;
}

std::vector<DrawObject> loadWavefront(const char* filename)
{
	std::vector<DrawObject> objects;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::map<std::string, GLuint> textures;
	float bmin[3], bmax[3];

	std::string base_dir = "";
	std::string obj_filename = std::string(filename);

	size_t last_slash = obj_filename.find_last_of("/\\");
	if (last_slash != std::string::npos)
		base_dir = obj_filename.substr(0, last_slash);

	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename,
			"", true);
	if (!err.empty())
	{
		fprintf(stderr, "%s\n", err.c_str());
	}

	if (!ret)
	{
		fprintf(stderr, "Failed to load %s\n", filename);
		exit(1);
	}

	fprintf(stderr, "# of vertices  = %d\n",
			(int) (attrib.vertices.size()) / 3);
	fprintf(stderr, "# of normals   = %d\n", (int) (attrib.normals.size()) / 3);
	fprintf(stderr, "# of texcoords = %d\n",
			(int) (attrib.texcoords.size()) / 2);
	fprintf(stderr, "# of materials = %d\n", (int) materials.size());
	fprintf(stderr, "# of shapes    = %d\n", (int) shapes.size());

	for (size_t m = 0; m < materials.size(); m++)
	{
		tinyobj::material_t* mp = &materials[m];

		if (mp->diffuse_texname.length() > 0)
		{
			// Only load the texture if it is not already loaded
			if (textures.find(mp->diffuse_texname) == textures.end())
			{
				GLuint texture_id;
				int w, h;
				int comp;

				const char* texture_filename = mp->diffuse_texname.c_str();
				if (!fileExists(texture_filename))
				{
					// Append base dir.
					texture_filename = (base_dir + mp->diffuse_texname).c_str();
					if (!fileExists(texture_filename))
					{
						fprintf(stderr, "Unable to find file: %s\n", mp->diffuse_texname.c_str());
						exit(1);
					}
				}

				textures[mp->diffuse_texname] = loadTexture(texture_filename);
			}
		}
	}

	// Append `default` material
	materials.push_back(tinyobj::material_t());

	bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

	for (size_t s = 0; s < shapes.size(); s++)
	{
		DrawObject o;
		std::vector<float> vb; // pos(3float), normal(3float), color(3float)
		for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++)
		{
			tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
			tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
			tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

			int current_material_id = shapes[s].mesh.material_ids[f];

			if ((current_material_id < 0)
					|| (current_material_id
							>= static_cast<int>(materials.size())))
			{
				// Invaid material ID. Use default material.
				current_material_id = materials.size() - 1; // Default material is added to the last item in `materials`.
			}

			float diffuse[3];
			for (size_t i = 0; i < 3; i++)
			{
				diffuse[i] = materials[current_material_id].diffuse[i];
			}
			float tc[3][2];
			if (attrib.texcoords.size() > 0
					&& idx0.texcoord_index > -1
					&& idx1.texcoord_index > -1
					&& idx2.texcoord_index > -1)
			{
				assert(attrib.texcoords.size() > 2 * idx0.texcoord_index + 1);
				assert(attrib.texcoords.size() > 2 * idx1.texcoord_index + 1);
				assert(attrib.texcoords.size() > 2 * idx2.texcoord_index + 1);
				tc[0][0] = attrib.texcoords[2 * idx0.texcoord_index];
				tc[0][1] = 1.0f - attrib.texcoords[2 * idx0.texcoord_index + 1];
				tc[1][0] = attrib.texcoords[2 * idx1.texcoord_index];
				tc[1][1] = 1.0f - attrib.texcoords[2 * idx1.texcoord_index + 1];
				tc[2][0] = attrib.texcoords[2 * idx2.texcoord_index];
				tc[2][1] = 1.0f - attrib.texcoords[2 * idx2.texcoord_index + 1];
			}
			else
			{
				tc[0][0] = 0.0f;
				tc[0][1] = 0.0f;
				tc[1][0] = 0.0f;
				tc[1][1] = 0.0f;
				tc[2][0] = 0.0f;
				tc[2][1] = 0.0f;
			}

			float v[3][3];
			for (int k = 0; k < 3; k++)
			{
				int f0 = idx0.vertex_index;
				int f1 = idx1.vertex_index;
				int f2 = idx2.vertex_index;
				assert(f0 >= 0);
				assert(f1 >= 0);
				assert(f2 >= 0);

				v[0][k] = attrib.vertices[3 * f0 + k];
				v[1][k] = attrib.vertices[3 * f1 + k];
				v[2][k] = attrib.vertices[3 * f2 + k];
				bmin[k] = std::min(v[0][k], bmin[k]);
				bmin[k] = std::min(v[1][k], bmin[k]);
				bmin[k] = std::min(v[2][k], bmin[k]);
				bmax[k] = std::max(v[0][k], bmax[k]);
				bmax[k] = std::max(v[1][k], bmax[k]);
				bmax[k] = std::max(v[2][k], bmax[k]);
			}

			float n[3][3];
			int f0 = idx0.normal_index;
			int f1 = idx1.normal_index;
			int f2 = idx2.normal_index;
			assert(f0 >= 0);
			assert(f1 >= 0);
			assert(f2 >= 0);
			for (int k = 0; k < 3; k++)
			{
				n[0][k] = attrib.normals[3 * f0 + k];
				n[1][k] = attrib.normals[3 * f1 + k];
				n[2][k] = attrib.normals[3 * f2 + k];
			}

			for (int k = 0; k < 3; k++)
			{
				vb.push_back(v[k][0]);
				vb.push_back(v[k][1]);
				vb.push_back(v[k][2]);
				vb.push_back(n[k][0]);
				vb.push_back(n[k][1]);
				vb.push_back(n[k][2]);
				// Combine normal and diffuse to get color.
				float normal_factor = 0.2;
				float diffuse_factor = 1 - normal_factor;
				float c[3] =
				{ n[k][0] * normal_factor + diffuse[0] * diffuse_factor, n[k][1]
						* normal_factor + diffuse[1] * diffuse_factor, n[k][2]
						* normal_factor + diffuse[2] * diffuse_factor };
				float len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
				if (len2 > 0.0f)
				{
					float len = sqrtf(len2);

					c[0] /= len;
					c[1] /= len;
					c[2] /= len;
				}
				vb.push_back(c[0] * 0.5 + 0.5);
				vb.push_back(c[1] * 0.5 + 0.5);
				vb.push_back(c[2] * 0.5 + 0.5);

				vb.push_back(tc[k][0]);
				vb.push_back(tc[k][1]);
			}
		}

		o.vb = 0;
		o.numTriangles = 0;

		if (shapes[s].mesh.material_ids.size() > 0
				&& shapes[s].mesh.material_ids.size() > s)
		{
			// Base case
			o.material_id = shapes[s].mesh.material_ids[s];
		}
		else
		{
			o.material_id = materials.size() - 1; // = ID for default material.
		}

		if (vb.size() > 0)
		{
			glTry(glGenBuffers(1, &o.vb));
			glTry(glBindBuffer(GL_ARRAY_BUFFER, o.vb));
			glTry(
					glBufferData(GL_ARRAY_BUFFER, vb.size() * sizeof(float), &vb.at(0), GL_STATIC_DRAW));
			o.numTriangles = vb.size() / (3 + 3 + 3 + 2) * 3;
			fprintf(stderr, "shape[%d] # of triangles = %d\n", static_cast<int>(s),
					o.numTriangles);
		}

		objects.push_back(o);
	}

	return objects;
}

