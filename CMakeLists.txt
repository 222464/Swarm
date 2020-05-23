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
    "${SOURCE_PATH}/Helpers.cpp"
    "${SOURCE_PATH}/Hierarchy.cpp"
    "${SOURCE_PATH}/LayerConv.cpp"
    "${SOURCE_PATH}/LayerPool.cpp"
    "${SOURCE_PATH}/OptimizerMAB.cpp"
)

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp

add_library(Swarm ${SOURCES})

target_link_libraries(Swarm ${OpenMP_CXX_LIBRARIES})

install(TARGETS Swarm
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include/swarm
        FILES_MATCHING PATTERN "*.h*")