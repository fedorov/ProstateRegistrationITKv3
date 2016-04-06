
#-----------------------------------------------------------------------------
project(ProstateRegistration)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(ProstateRegistration ProstateRegistration.cxx)
target_link_libraries(ProstateRegistration ${ITK_LIBRARIES})
