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
    "${SOURCE_PATH}/swarm/SparseCoder.cpp"
    "${SOURCE_PATH}/swarm/Predictor.cpp"
    "${SOURCE_PATH}/swarm/Actor.cpp"
    "${SOURCE_PATH}/swarm/Hierarchy.cpp"
    "${SOURCE_PATH}/swarm/ImageEncoder.cpp"
	"${SOURCE_PATH}/swarm/SparseMatrix.cpp"
)

set(HEADERS
	"${SOURCE_PATH}/swarm/Helpers.h"
    "${SOURCE_PATH}/swarm/SparseCoder.h"
    "${SOURCE_PATH}/swarm/Predictor.h"
    "${SOURCE_PATH}/swarm/Actor.h"
    "${SOURCE_PATH}/swarm/Hierarchy.h"
    "${SOURCE_PATH}/swarm/ImageEncoder.h"
	"${SOURCE_PATH}/swarm/SparseMatrix.h"
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