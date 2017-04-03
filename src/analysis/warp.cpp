#include "glutils.h"

#include <stdio.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/quaternion.hpp>

#include <vector>

using namespace glm;

GLuint shaderprog;
vec3 camOrigin;
vec3 proj1Origin, proj2Origin;
std::vector<DrawObject> objects;

GLuint loadShaders()
{
	// Create the shaders
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the shader code from file
	char* vertShaderCode = file2buf("projection.vert");
	char* fragShaderCode = file2buf("projection.frag");

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

void loadTextures()
{
	int loc;
	GLuint id;

	glActiveTexture(GL_TEXTURE0);
	id = loadTexture("0001.png");
	glBindTexture(GL_TEXTURE_2D, id);

	loc = glGetUniformLocation(shaderprog, "Proj1Tex");
	glTry(glUniform1i(loc, 0));

	glActiveTexture(GL_TEXTURE1);
	id = loadTexture("0002.png");
	glBindTexture(GL_TEXTURE_2D, id);

	loc = glGetUniformLocation(shaderprog, "Proj2Tex");
	glTry(glUniform1i(loc, 1));
}

void loadObjects()
{
	std::vector<DrawObject> newObjects = loadWavefront("house.obj");
	objects.insert(objects.end(), newObjects.begin(), newObjects.end());
}

void projectTexture()
{
	double fov = 0.85756f; // 35mm film in radians
	double aspect = 1.0f;
	double near = 2.0f;
	double far = 2000.0f;
	int loc;

	vec3 target = vec3(0.0f, 0.0f, 0.0f);
	vec3 up = vec3(0.0f, 1.0f, 0.0f);

	// Projector 1
	mat4 proj1View = lookAt(proj1Origin, target, up);
	mat4 proj1Proj = perspective(fov, aspect, near, far);

	mat4 proj1MVP = proj1Proj * proj1View;
	proj1MVP = scale(proj1MVP, vec3(0.5f));
	proj1MVP = translate(proj1MVP, vec3(0.5f));

	loc = glGetUniformLocation(shaderprog, "Projector1MVP");
	glTry(glUniformMatrix4fv(loc, 1, GL_FALSE, &proj1MVP[0][0]));

	// Projector 2
	mat4 proj2View = lookAt(proj2Origin, target, up);
	mat4 proj2Proj = proj1Proj;

	mat4 proj2MVP = proj2Proj * proj2View;
	proj2MVP = scale(proj2MVP, vec3(0.5f));
	proj2MVP = translate(proj2MVP, vec3(0.5f));

	loc = glGetUniformLocation(shaderprog, "Projector2MVP");
	glTry(glUniformMatrix4fv(loc, 1, GL_FALSE, &proj2MVP[0][0]));
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
	glTry(glUniformMatrix4fv(loc, 1, GL_FALSE, &camMVP[0][0]));
}

void render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setCamera();
	projectTexture();

	for (size_t i = 0; i < objects.size(); i++)
	{
		DrawObject o = objects[i];
		if (o.vb < 1)
			continue;

		glBindBuffer(GL_ARRAY_BUFFER, o.vb);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		glVertexPointer(o.vertex_size, GL_FLOAT, o.stride, o.vertex_ptr);
		glNormalPointer(GL_FLOAT, o.stride, o.normal_ptr);
		glColorPointer(o.color_size, GL_FLOAT, o.stride, o.color_ptr);
		glTexCoordPointer(o.uv_size, GL_FLOAT, o.stride, o.uv_ptr);

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

		case GLFW_KEY_UP:
			proj1Origin += vec3(0.0f, 0.1f, 0.0f) * length(camOrigin);
			break;
		case GLFW_KEY_LEFT:
			proj1Origin = rotate(proj1Origin, -0.1f, vec3(0.0f, 1.0f, 0.0f));
			break;
		case GLFW_KEY_DOWN:
			proj1Origin -= vec3(0.0f, 0.1f, 0.0f) * length(camOrigin);
			break;
		case GLFW_KEY_RIGHT:
			proj1Origin = rotate(proj1Origin, 0.1f, vec3(0.0f, 1.0f, 0.0f));
			break;

		case GLFW_KEY_PAGE_UP:
			proj1Origin /= 2;
			break;
		case GLFW_KEY_PAGE_DOWN:
			proj1Origin *= 2;
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
	window = glfwCreateWindow(640, 640, "Warp Preview", NULL, NULL);
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
	loadTextures();
	camOrigin = vec3(-510.0f, 100.0f, -620.0f);
	proj1Origin = vec3(-420.0f, 100.0f, 45.0f);
	proj2Origin = vec3(0.0f, 100.0f, -580.0f);

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
