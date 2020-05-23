cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Swarm)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories("${PROJECT_SOURCE_DIR}/source")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/swarm/Helpers.cpp"
    "${SOURCE_PATH}/swarm/Hierarchy.cpp"
    "${SOURCE_PATH}/swarm/LayerConv.cpp"
	"${SOURCE_PATH}/swarm/LayerPool.cpp"
	"${SOURCE_PATH}/swarm/OptimizerMAB.cpp"
)

set(HEADERS
    "${SOURCE_PATH}/swarm/ComputeSystem.h"
	"${SOURCE_PATH}/swarm/Helpers.h"
    "${SOURCE_PATH}/swarm/Hierarchy.h"
    "${SOURCE_PATH}/swarm/Layer.h"
    "${SOURCE_PATH}/swarm/LayerConv.h"
	"${SOURCE_PATH}/swarm/LayerPool.h"
	"${SOURCE_PATH}/swarm/OptimizerMAB.h"
)

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp

add_library(Swarm ${SOURCES} ${HEADERS})

target_link_libraries(Swarm ${OpenMP_CXX_LIBRARIES})

install(TARGETS Swarm
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")