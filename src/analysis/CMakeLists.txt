find_package(OpenGL REQUIRED)
find_package(GLM REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(warp warp.cpp glutils.cpp)
target_link_libraries(warp
	PRIVATE GL
	PRIVATE GLEW
	PRIVATE glfw)

add_executable(capture capture.cpp glutils.cpp)
target_link_libraries(capture
	PRIVATE GL
	PRIVATE GLEW
	PRIVATE glfw)
	
add_subdirectory(depth)
