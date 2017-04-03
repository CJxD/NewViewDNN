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

// Function to check OpenGL error status
#define glTry(func) func; _check(__FILE__, __LINE__, #func)
void _check(const char* file, int line, const char* where);

// Function to check OpenGL shader error status
#define checkShader(shader) _checkShader(shader, __FILE__, __LINE__)
void _checkShader(GLuint shader, const char* file, int line);

// Function to check OpenGL program error status
#define checkProgram(program) _checkProgram(program, __FILE__, __LINE__)
void _checkProgram(GLuint program, const char* file, int line);

// A simple function that will read a file into an allocated char pointer buffer
char* filetobuf(const char* filename);

// Loads an image into the currently bound OpenGL texture buffer
void loadTexture(const char* filename);
