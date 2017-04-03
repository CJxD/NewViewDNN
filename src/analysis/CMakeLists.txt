find_package(OpenGL REQUIRED)
find_package(GLM REQUIRED)
find_package(glfw3 REQUIRED)
find_package(tinyobjloader REQUIRED)

add_executable(warp warp.cpp glutils.cpp)
target_link_libraries(warp
	PRIVATE GL
	PRIVATE GLEW
	PRIVATE glfw
	PRIVATE tinyobjloader)

add_subdirectory(depth)
