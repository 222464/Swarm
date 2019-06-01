cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Swarm)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

if(DISABLE_THREADING)
    message("Threading disabled")
    add_compile_definitions(KERNEL_NOTHREAD)
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
    "${SOURCE_PATH}/OptimizerMeta.cpp"
)

add_library(Swarm ${SOURCES})

install(TARGETS Swarm
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include/swarm
        FILES_MATCHING PATTERN "*.h*")