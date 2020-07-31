#####################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
# Enable C++17
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(HEADER_FILES Socket.cpp Socket.h MPU6050.cpp MPU6050.h)
FIND_LIBRARY(wiringPi_LIB wiringPi)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

# Link
ADD_EXECUTABLE(client main.cpp ${HEADER_FILES})

TARGET_LINK_LIBRARIES(client -lstdc++fs)
TARGET_LINK_LIBRARIES(client ${wiringPi_LIB})
TARGET_LINK_LIBRARIES(client ${Boost_LIBRARIES})
#####################################