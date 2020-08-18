#####################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
# Enable C++17
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(HEADER_FILES Socket.cpp Socket.h MPU6050.cpp MPU6050.h)
FIND_PACKAGE(OpenCV)
FIND_LIBRARY(wiringPi_LIB wiringPi)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(raspicam REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Threads
SET(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_HAVE_THREADS_LIBRARY 1)
SET(CMAKE_USE_WIN32_THREADS_INIT 0)
SET(CMAKE_USE_PTHREADS_INIT 1)
SET(THREADS_PREFER_PTHREAD_FLAG ON)

# Link
ADD_EXECUTABLE(client client.cpp ${HEADER_FILES})
ADD_EXECUTABLE(server server.cpp ${HEADER_FILES})
ADD_EXECUTABLE(cv_client cv_client.cpp ${HEADER_FILES})
ADD_EXECUTABLE(cv_server cv_server.cpp ${HEADER_FILES})

TARGET_LINK_LIBRARIES(client -lstdc++fs ${wiringPi_LIB} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(server -lstdc++fs ${wiringPi_LIB} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(cv_client ${wiringPi_LIB} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${raspicam_CV_LIBS} -lstdc++fs)
TARGET_LINK_LIBRARIES(cv_server ${wiringPi_LIB} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${raspicam_CV_LIBS} -lstdc++fs)
#####################################