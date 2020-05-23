# Locate Swarm library
#
# This module defines
# SWARM_LIBRARY, the name of the library to link against
# SWARM_FOUND, if false, do not try to link to Swarm
# SWARM_INCLUDE_DIR, where to find Swarm headers
#
if(SWARM_INCLUDE_DIR)
  # Already in cache, be silent
  set(SWARM_FIND_QUIETLY true)
endif(SWARM_INCLUDE_DIR)

find_path(SWARM_INCLUDE_DIR swarm/Hierarchy.h)
if (not "${SWARM_INCLUDE_DIR}" strequal "SWARM_INCLUDE_DIR-NOTFOUND")
  set(SWARM_INCLUDE_DIR "${SWARM_INCLUDE_DIR}/swarm")
endif()

set(SWARM_NAMES swarm Swarm SWARM)
find_library()(SWARM_LIBRARY NAMES ${SWARM_NAMES})

# Per-recommendation
set(SWARM_INCLUDE_DIRS "${SWARM_INCLUDE_DIR}")
set(SWARM_LIBRARIES    "${SWARM_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set SWARM_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Swarm DEFAULT_MSG SWARM_LIBRARY SWARM_INCLUDE_DIR)