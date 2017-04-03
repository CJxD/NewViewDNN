#find_package(orb_slam REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(../../../deps/orb_slam)
include_directories(../../../deps/orb_slam/include)

#add_executable(extract_pointcloud extract_pointcloud.cpp)
#target_link_libraries(extract_pointcloud ${OpenCV_LIBS} ${OpenGL_LIBS} ${Pangolin_LIBS})
