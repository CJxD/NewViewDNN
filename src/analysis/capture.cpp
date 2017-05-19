#include "glutils.h"

#include <stdio.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/quaternion.hpp>

#include <vector>

using namespace glm;

struct Light
{
	vec3 position;
	float intensity;
};

GLuint shaderprog;
vec3 camOrigin;
std::vector<DrawObject> objects;
std::vector<Light> lights;

GLuint loadShaders()
{
	// Create the shaders
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the shader code from file
	char* vertShaderCode = file2buf("solid.vert");
	char* fragShaderCode = file2buf("blinn-phong.frag");

	// Compile Vertex Shader
	glShaderSource(vertShader, 1, &vertShaderCode, NULL);
	glCompileShader(vertShader);

	// Compile Fragment Shader
	glShaderSource(fragShader, 1, &fragShaderCode, NULL);
	glCompileShader(fragShader);

	free(vertShaderCode);
	free(fragShaderCode);
	checkShader(vertShader);
	checkShader(fragShader);

	// Link the program
	fprintf(stderr, "Linking shader program\n");
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vertShader);
	glAttachShader(prog, fragShader);
	glLinkProgram(prog);
	checkProgram(prog);

	glDetachShader(prog, vertShader);
	glDetachShader(prog, fragShader);

	glDeleteShader(vertShader);
	glDeleteShader(fragShader);

	return prog;
}

void loadObjects()
{
	std::vector<DrawObject> newObjects = loadWavefront("house.obj");
	objects.insert(objects.end(), newObjects.begin(), newObjects.end());
}

void bindBuffers(DrawObject &o)
{
	glBindBuffer(GL_ARRAY_BUFFER, o.vb);

	// Vertex attrib pointer
	glVertexAttribPointer(0, o.vertex_size, GL_FLOAT, GL_FALSE, o.stride,
			o.vertex_ptr);
	glEnableVertexAttribArray(0);

	// Normal attrib pointer
	glVertexAttribPointer(1, o.normal_size, GL_FLOAT, GL_FALSE, o.stride,
			o.normal_ptr);
	glEnableVertexAttribArray(1);

	// UV coordinates attrib pointer
	glVertexAttribPointer(2, o.uv_size, GL_FLOAT, GL_FALSE, o.stride, o.uv_ptr);
	glEnableVertexAttribArray(2);
}

void bindMaterial(DrawObject::Material &m)
{
	int loc;

	loc = glGetUniformLocation(shaderprog, "material.ambient");
	glTry(glUniform3fv(loc, 1, m.ambient));

	loc = glGetUniformLocation(shaderprog, "material.diffuse");
	glTry(glUniform3fv(loc, 1, m.diffuse));

	loc = glGetUniformLocation(shaderprog, "material.specular");
	glTry(glUniform3fv(loc, 1, m.specular));

	loc = glGetUniformLocation(shaderprog, "material.mode");
	glTry(glUniform1ui(loc, 0));

	loc = glGetUniformLocation(shaderprog, "material.hasAmbientTex");
	glTry(glUniform1ui(loc, m.ambientTex > 0));

	loc = glGetUniformLocation(shaderprog, "material.hasDiffuseTex");
	glTry(glUniform1ui(loc, m.diffuseTex > 0));

	loc = glGetUniformLocation(shaderprog, "material.hasSpecularTex");
	glTry(glUniform1ui(loc, m.specularTex > 0));

	loc = glGetUniformLocation(shaderprog, "material.hasBumpTex");
	glTry(glUniform1ui(loc, m.bumpTex > 0));

	if (m.ambientTex > 0)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m.ambientTex);
		loc = glGetUniformLocation(shaderprog, "AmbientTex");
		glTry(glUniform1i(loc, 0));
	}

	if (m.diffuseTex > 0)
	{
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, m.diffuseTex);
		loc = glGetUniformLocation(shaderprog, "DiffuseTex");
		glTry(glUniform1i(loc, 1));
	}

	if (m.specularTex > 0)
	{
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, m.specularTex);
		loc = glGetUniformLocation(shaderprog, "SpecularTex");
		glTry(glUniform1i(loc, 2));
	}

	if (m.bumpTex > 0)
	{
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, m.bumpTex);
		loc = glGetUniformLocation(shaderprog, "BumpTex");
		glTry(glUniform1i(loc, 3));
	}
}

void setCamera()
{
	double fov = 0.85756f; // 35mm film in radians
	double aspect = 1.0f;
	double near = 2.0f;
	double far = 2000.0f;

	vec3 target = vec3(0.0f, 0.0f, 0.0f);
	vec3 up = vec3(0.0f, 1.0f, 0.0f);

	mat4 camView = lookAt(camOrigin, target, up);
	mat4 camProj = perspective(fov, aspect, near, far);

	mat4 camMVP = camProj * camView;

	int loc = glGetUniformLocation(shaderprog, "CameraMVP");
	glTry(glUniformMatrix4fv(loc, 1, GL_FALSE, value_ptr(camMVP)));
}

void render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setCamera();

	for (DrawObject o : objects)
	{
		DrawObject::Material& m = o.material;
		if (o.vb < 1)
			continue;

		bindBuffers(o);
		bindMaterial(m);

		glTry(glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles));
	}
}

void setupGL()
{
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glClearColor(0.9, 0.9, 0.9, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	shaderprog = loadShaders();
	glTry(glUseProgram(shaderprog));
}

void onKeyPress(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS || action == GLFW_REPEAT)
	{
		switch (key)
		{
		case GLFW_KEY_ESCAPE:
		case GLFW_KEY_Q:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_W:
			camOrigin += vec3(0.0f, 0.1f, 0.0f) * length(camOrigin);
			break;
		case GLFW_KEY_A:
			camOrigin = rotate(camOrigin, -0.1f, vec3(0.0f, 1.0f, 0.0f));
			break;
		case GLFW_KEY_S:
			camOrigin -= vec3(0.0f, 0.1f, 0.0f) * length(camOrigin);
			break;
		case GLFW_KEY_D:
			camOrigin = rotate(camOrigin, 0.1f, vec3(0.0f, 1.0f, 0.0f));
			break;
		}
	}
}

int main()
{
	// Setup window
	GLFWwindow* window;
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to start GLFW\n");
		return EXIT_FAILURE;
	}
	window = glfwCreateWindow(640, 640, "Capture Preview", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		fprintf(stderr, "GLFW Failed to start\n");
		return EXIT_FAILURE;
	}

	// Link window to GLEW
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	int err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}
	fprintf(stderr, "GLEW done\n");
	glfwSetKeyCallback(window, onKeyPress);
	fprintf(stderr, "GL INFO %s\n", glGetString(GL_VERSION));

	setupGL();
	loadObjects();
	camOrigin = vec3(-510.0f, 100.0f, -620.0f);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		render();

		glfwSwapBuffers(window); // Swap front and back rendering buffers
		glfwPollEvents(); // Poll for events.
	}

	glfwTerminate();
	exit(EXIT_SUCCESS);
}
