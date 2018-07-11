cmake_minimum_required(VERSION 2.8.3)
project(SqueezeSeg)

set(PYTHON_FILES src)

add_executable(SqueezeSeg_tf ${PYTHON_FILES})
#target_link_libraries(SqueezeSeg_tf ${PYTHON_LIBRARIES})