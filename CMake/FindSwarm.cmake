# Locate Swarm library
#
# This module defines
# SWARM_LIBRARY, the name of the library to link against
# SWARM_FOUND, if false, do not try to link to Swarm
# SWARM_INCLUDE_DIR, where to find Swarm headers
#
IF(SWARM_INCLUDE_DIR)
  # Already in cache, be silent
  SET(SWARM_FIND_QUIETLY TRUE)
ENDIF(SWARM_INCLUDE_DIR)

FIND_PATH(SWARM_INCLUDE_DIR swarm/Hierarchy.h)
if (NOT "${SWARM_INCLUDE_DIR}" STREQUAL "SWARM_INCLUDE_DIR-NOTFOUND")
  SET(SWARM_INCLUDE_DIR "${SWARM_INCLUDE_DIR}/swarm")
ENDIF()

SET(SWARM_NAMES swarm Swarm SWARM)
FIND_LIBRARY(SWARM_LIBRARY NAMES ${SWARM_NAMES})

# Per-recommendation
SET(SWARM_INCLUDE_DIRS "${SWARM_INCLUDE_DIR}")
SET(SWARM_LIBRARIES    "${SWARM_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set SWARM_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Swarm DEFAULT_MSG SWARM_LIBRARY SWARM_INCLUDE_DIR)