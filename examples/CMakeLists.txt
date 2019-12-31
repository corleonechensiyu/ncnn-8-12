
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
find_package(OpenCV REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)


#link_libraries(pthread)
#link_libraries(wiringPi)
set(NCNN_EXAMPLE_LINK_LIBRARIES ncnn ${OpenCV_LIBS})
if(NCNN_VULKAN)
    list(APPEND NCNN_EXAMPLE_LINK_LIBRARIES ${Vulkan_LIBRARY})
endif()

#add_executable(squeezenet squeezenet.cpp)
#target_link_libraries(squeezenet ${NCNN_EXAMPLE_LINK_LIBRARIES})

#add_executable(absdiff absdiff.cpp)
#target_link_libraries(absdiff ${NCNN_EXAMPLE_LINK_LIBRARIES})

#add_executable(picture picture.cpp)
#target_link_libraries(picture ${NCNN_EXAMPLE_LINK_LIBRARIES})

#add_executable(video vedio.cpp)
#target_link_libraries(video ${NCNN_EXAMPLE_LINK_LIBRARIES})

#add_executable(ocrssd ocrssd.cpp)
#target_link_libraries(ocrssd ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(test_permute test_permute.cpp)
target_link_libraries(test_permute ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(dot_python dot_python.cpp)
target_link_libraries(dot_python ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(textssd textssd.cpp)
target_link_libraries(textssd ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(AdvancedEast AdvancedEast.cpp)
target_link_libraries(AdvancedEast ${NCNN_EXAMPLE_LINK_LIBRARIES})

#add_executable(shufflenetssd shufflenetssd.cpp)
#target_link_libraries(shufflenetssd ${NCNN_EXAMPLE_LINK_LIBRARIES})
