# # cmake_minimum_required(VERSION 3.10)
# # project(EmotionRecognition)

# # find_package(OpenCV REQUIRED)
# # include_directories(${OpenCV_INCLUDE_DIRS})

# # add_executable(EmotionRecognition test2.cpp)
# # target_link_libraries(EmotionRecognition ${OpenCV_LIBS})

# cmake_minimum_required(VERSION 3.10)
# project(EmotionRecognition)

# set(CMAKE_CXX_STANDARD 17)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(EmotionRecognition test2.cpp)
# target_link_libraries(EmotionRecognition ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.10)
project(EmotionRecognition)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include the Dlib installation path
set(DLIB_INCLUDE_DIR /opt/homebrew/Cellar/dlib/19.24.6/include)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include Dlib
include_directories(${DLIB_INCLUDE_DIR})

# Manually specify the path for OpenBLAS
set(OPENBLAS_DIR /opt/homebrew/Cellar/openblas/0.3.28) # Update this path if needed
set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include)
set(OPENBLAS_LIBRARY ${OPENBLAS_DIR}/lib/libopenblasp-r0.3.28.dylib)

# Include OpenBLAS
include_directories(${OPENBLAS_INCLUDE_DIR})

# Add the source files for your project
add_executable(EmotionRecognition test2.cpp) # Replace with your actual source file

# Link the necessary libraries
target_link_libraries(EmotionRecognition ${OpenCV_LIBS} ${DLIB_INCLUDE_DIR} ${OPENBLAS_LIBRARY})

# Print the paths for debugging purposes
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Dlib include directory: ${DLIB_INCLUDE_DIR}")
message(STATUS "OpenBLAS library: ${OPENBLAS_LIBRARY}")
