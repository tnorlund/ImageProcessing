#####################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
# Enable C++17
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
FIND_LIBRARY(wiringPi_LIB wiringPi)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

# Link
ADD_EXECUTABLE(client main.cpp)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(client -lstdc++fs)
TARGET_LINK_LIBRARIES(client ${wiringPi_LIB})
TARGET_LINK_LIBRARIES(client ${Boost_LIBRARIES})
#####################################