#include "warp.h"
#include "glutils.h"

#include <stdio.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/quaternion.hpp>

#include <tiny_obj_loader.h>

using namespace glm;

typedef struct {
	GLuint vb; // Vertex buffer
	int numTriangles;
	size_t material_id;
} DrawObject;

GLuint shaderprog;
vec3 camOrigin;
vec3 proj1Origin, proj2Origin;
std::vector<DrawObject> objects;

void loadObject(const char* filename) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename,
			"", true);
	if (!err.empty()) {
		fprintf(stderr, "%s\n", err.c_str());
	}

	if (!ret) {
		fprintf(stderr, "Failed to load %s\n", filename);
		exit(1);
	}

	printf("# of vertices  = %d\n", (int) (attrib.vertices.size()) / 3);
	printf("# of normals   = %d\n", (int) (attrib.normals.size()) / 3);
	printf("# of texcoords = %d\n", (int) (attrib.texcoords.size()) / 2);
	printf("# of materials = %d\n", (int) materials.size());
	printf("# of shapes    = %d\n", (int) shapes.size());

	// Append `default` material
	materials.push_back(tinyobj::material_t());

	for (size_t s = 0; s < shapes.size(); s++) {
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
							>= static_cast<int>(materials.size()))) {
				// Invaid material ID. Use default material.
				current_material_id = materials.size() - 1; // Default material is added to the last item in `materials`.
			}

			float diffuse[3];
			for (size_t i = 0; i < 3; i++) {
				diffuse[i] = materials[current_material_id].diffuse[i];
			}
			float tc[3][2];
			/*if (attrib.texcoords.size() > 0) {
				assert(
						attrib.texcoords.size()
								> 2 * idx0.texcoord_index + 1);
				assert(
						attrib.texcoords.size()
								> 2 * idx1.texcoord_index + 1);
				assert(
						attrib.texcoords.size()
								> 2 * idx2.texcoord_index + 1);
				tc[0][0] = attrib.texcoords[2 * idx0.texcoord_index];
				tc[0][1] = 1.0f
						- attrib.texcoords[2 * idx0.texcoord_index + 1];
				tc[1][0] = attrib.texcoords[2 * idx1.texcoord_index];
				tc[1][1] = 1.0f
						- attrib.texcoords[2 * idx1.texcoord_index + 1];
				tc[2][0] = attrib.texcoords[2 * idx2.texcoord_index];
				tc[2][1] = 1.0f
						- attrib.texcoords[2 * idx2.texcoord_index + 1];
			} else {*/
				tc[0][0] = 0.0f;
				tc[0][1] = 0.0f;
				tc[1][0] = 0.0f;
				tc[1][1] = 0.0f;
				tc[2][0] = 0.0f;
				tc[2][1] = 0.0f;
			//}

			float v[3][3];
			for (int k = 0; k < 3; k++) {
				int f0 = idx0.vertex_index;
				int f1 = idx1.vertex_index;
				int f2 = idx2.vertex_index;
				assert(f0 >= 0);
				assert(f1 >= 0);
				assert(f2 >= 0);

				v[0][k] = attrib.vertices[3 * f0 + k];
				v[1][k] = attrib.vertices[3 * f1 + k];
				v[2][k] = attrib.vertices[3 * f2 + k];
			}

			float n[3][3];
			int f0 = idx0.normal_index;
			int f1 = idx1.normal_index;
			int f2 = idx2.normal_index;
			assert(f0 >= 0);
			assert(f1 >= 0);
			assert(f2 >= 0);
			for (int k = 0; k < 3; k++) {
				n[0][k] = attrib.normals[3 * f0 + k];
				n[1][k] = attrib.normals[3 * f1 + k];
				n[2][k] = attrib.normals[3 * f2 + k];
			}

			for (int k = 0; k < 3; k++) {
				vb.push_back(v[k][0]);
				vb.push_back(v[k][1]);
				vb.push_back(v[k][2]);
				vb.push_back(n[k][0]);
				vb.push_back(n[k][1]);
				vb.push_back(n[k][2]);
				// Combine normal and diffuse to get color.
				float normal_factor = 0.2;
				float diffuse_factor = 1 - normal_factor;
				float c[3] = { n[k][0] * normal_factor
						+ diffuse[0] * diffuse_factor, n[k][1]
						* normal_factor + diffuse[1] * diffuse_factor,
						n[k][2] * normal_factor
								+ diffuse[2] * diffuse_factor };
				float len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
				if (len2 > 0.0f) {
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
				&& shapes[s].mesh.material_ids.size() > s) {
			// Base case
			o.material_id = shapes[s].mesh.material_ids[s];
		} else {
			o.material_id = materials.size() - 1; // = ID for default material.
		}

		if (vb.size() > 0) {
			glTry(glGenBuffers(1, &o.vb));
			glTry(glBindBuffer(GL_ARRAY_BUFFER, o.vb));
			glTry(glBufferData(GL_ARRAY_BUFFER, vb.size() * sizeof(float),
					&vb.at(0),
					GL_STATIC_DRAW));
			o.numTriangles = vb.size() / (3 + 3 + 3 + 2) * 3;
			printf("shape[%d] # of triangles = %d\n", static_cast<int>(s),
					o.numTriangles);
		}

		objects.push_back(o);
	}
}

GLuint loadShaders() {
	// Create the shaders
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the shader code from file
	char* vertShaderCode = filetobuf("projection.vert");
	char* fragShaderCode = filetobuf("projection.frag");

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

void loadTextures() {
	int loc;
	GLuint textureIDs[2];
	glGenTextures(2, textureIDs);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureIDs[0]);
	loadTexture("0001.png");

	loc = glGetUniformLocation(shaderprog, "Proj1Tex");
	glTry(glUniform1i(loc, 0));

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, textureIDs[1]);
	loadTexture("0002.png");

	loc = glGetUniformLocation(shaderprog, "Proj2Tex");
	glTry(glUniform1i(loc, 1));
}

void projectTexture() {
	double fov = 0.85756f; // 35mm film in radians
	double aspect = 1.0f;
	double near = 2.0f;
	double far = 2000.0f;
	int loc;

	vec3 target = vec3(0.0f, 0.0f, 0.0f);
	vec3 up = vec3(0.0f, 1.0f, 0.0f);

	// Projector 1
	//mat4 proj1View = lookAt(proj1Origin, target, up);
	mat4 rotation = (mat4) quat(-0.51f, -0.5f, 0.49f, 0.499f);
	mat4 proj1View = transpose(translate(proj1Origin)) * transpose(rotation);
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

void setCamera() {
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

void render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	setCamera();
	projectTexture();

	GLsizei stride = (3 + 3 + 3 + 2) * sizeof(float);

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

		glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
		glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
		glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));
		glTexCoordPointer(2, GL_FLOAT, stride, (const void*)(sizeof(float) * 9));

		glTry(glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles));
	}
}

void setupGL() {
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glClearColor(0.9, 0.9, 0.9, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	shaderprog = loadShaders();
	glTry(glUseProgram(shaderprog));
}

void onKeyPress(GLFWwindow* window, int key, int scancode, int action,
		int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
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

int main() {
	// Setup window
	GLFWwindow* window;
	if (!glfwInit()) {
		fprintf(stderr, "Failed to start GLFW\n");
		return EXIT_FAILURE;
	}
	window = glfwCreateWindow(640, 640, "Warp Preview", NULL, NULL);
	if (!window) {
		glfwTerminate();
		fprintf(stderr, "GLFW Failed to start\n");
		return EXIT_FAILURE;
	}

	// Link window to GLEW
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	int err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}
	fprintf(stderr, "GLEW done\n");
	glfwSetKeyCallback(window, onKeyPress);
	fprintf(stderr, "GL INFO %s\n", glGetString(GL_VERSION));

	setupGL();
	loadObject("house.obj");
	loadTextures();
	camOrigin = vec3(-510.0f, 100.0f, -620.0f);
	proj1Origin = vec3(-420.0f, 100.0f, 45.0f);
	proj2Origin = vec3(0.0f, 100.0f, -580.0f);

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		render();

		glfwSwapBuffers(window); // Swap front and back rendering buffers
		glfwPollEvents(); // Poll for events.
	}

	glfwTerminate();
	exit(EXIT_SUCCESS);
}
