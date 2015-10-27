cmake_minimum_required(VERSION 2.8.3)
project(appl_online)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable")

include_directories(
  src
)

add_library("${PROJECT_NAME}" SHARED
  src/random_streams.cpp
  src/globals.cpp
  src/problems/lasertag/lasertag.cpp
  src/problems/tag/belief_update_tag.cpp
  src/problems/tag/tag.cpp
  src/problems/bridge/bridge.cpp
  src/problems/tiger/tiger.cpp
  src/problems/rocksample/rocksample.cpp
  src/problems/pocman/pocman.cpp
  src/problems/pocman/coord.cpp
)

add_executable("${PROJECT_NAME}_despot" src/main.cpp)
target_link_libraries("${PROJECT_NAME}_despot" "${PROJECT_NAME}")

install(TARGETS "${PROJECT_NAME}" "${PROJECT_NAME}_despot"
  EXPORT "${PROJECT_NAME}Targets"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY src/
  DESTINATION include/appl/online
  FILES_MATCHING PATTERN "*.h"
)
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)
