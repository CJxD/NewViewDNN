#include "glutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void _check(const char* file, int line, const char* where) {
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
	fprintf(stderr, "Error (%d) %s in %s line %d\n%s\n", err, what, file, line, where);
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

		fprintf(stderr, "Shader compile error in %s line %d:\n%s\n", file, line, &err[0]);
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

		fprintf(stderr, "Program link error in %s line %d:\n%s\n", file, line, &err[0]);
		exit(1);
	}
}

char* filetobuf(const char* filename)
{
	FILE *fptr;
	long length;
	char *buf;
	fprintf(stderr, "Loading %s\n", filename);
	fptr = fopen(filename, "rb"); /* Open file for reading */
	if (!fptr) { /* Return NULL on failure */
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

void loadTexture(const char* filename)
{
	int width, height;
	int comp;

	fprintf(stderr, "Loading %s\n", filename);
	stbi_set_flip_vertically_on_load(true);
	unsigned char* image = stbi_load(filename, &width, &height, &comp, STBI_default);
	if (!image) {
		fprintf(stderr, "Unable to load texture: %s\n", filename);
		exit(1);
	}

	if (comp == 3)
	{
		glTry(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image));
	}
	else if (comp == 4)
	{
		glTry(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image));
	}

	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
	glTry(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));

	stbi_image_free(image);
}
