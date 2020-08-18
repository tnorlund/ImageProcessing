/**
 *   This takes a picture with the connected camera and sends it to the server.
 */

#include <iostream>
#include <exception>
#include <string>
#include <regex>

#include <fstream>
#include <typeinfo>

#include "MPU6050.h"
#include "Socket.h"
#include <boost/program_options.hpp>

#include <raspicam/raspicam_cv.h>
#include <raspicam/raspicam_still_cv.h>
#include <opencv2/opencv.hpp>

#define IMG_HEIGHT  960
#define IMG_WIDTH   1280

/**
 *   The size of the received packet.
 */
const int RCVBUFFERSIZE = 32;

/**
 *   Checks the Raspberry Pi for a connected camera.
 * 
 *   This function checks the Raspberry Pi for whether the system is camera-
 *   compatible and if a camera is connected.
 */
void checkCameraConfiguration() {
  FILE *fp;
  char var[23];
  std::string cameraString;
  std::smatch cameraMatch;
  std::regex cameraRegex("supported=(0|1) detected=(0|1)");

  fp = popen("/opt/vc/bin/vcgencmd get_camera", "r");
  fgets(var, sizeof(var), fp);
  pclose(fp);
  cameraString = std::string(var);
  if (
    std::regex_search(cameraString, cameraMatch, cameraRegex)
  ) {
    if (std::string(cameraMatch[1]) != "1")
      throw "Camera is not supported";
    if (std::string(cameraMatch[2]) != "1")
      throw "Camera is not connected";
  } else {
    throw "Unable to get camera information";
  }
}

/**
 *   Handle the parameters given in the command line.
 * 
 *   @param server the ip of the server
 *   @param port the port used to connect to the server
 *   @returns whether the parameters passed are proper or not
 */
bool processCommandLine(int argc, char** argv,
                          std::string& server,
                          std::string& port) {
  try {
    boost::program_options::options_description description("Allowed options");
    description.add_options()
      ("help,h", "Help screen")
      ("server,s", boost::program_options::value(&server)->required(), 
        "the server IP address")
      ("port,p", boost::program_options::value(&port)->required(), 
        "the port used to connect to the server");
    
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv)
            .options(description).run (), 
        vm
    );

    if (vm.count("help")) {
      std::cout << description << "\n";
      return false;
    }

    boost::program_options::notify(vm);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return false;
  } catch(...) {
    std::cerr << "Unknown error!" << "\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  bool result;
  std::string server, port;
  unsigned short clientPort;
  int imageSize;
  char *buf = (char *)malloc(IMG_HEIGHT * IMG_WIDTH * 3);
  cv::Mat frame(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, buf);
  raspicam::RaspiCam_Cv Camera;

  // Handle the arguments passed via command line
  result = processCommandLine(argc, argv, server, port);
  if (!result)
    return 1;

  // Check to see if the camera is connected an accesible.
  checkCameraConfiguration();

  clientPort = std::stoi(port);

  
  if ( !Camera.open() ) { throw "Error opening the camera"; }
  Camera.grab();
  Camera.retrieve(frame);
  Camera.release();
  cv::imwrite("Tyler_Norlund_client.png", frame);
  frame = (frame.reshape(0,1));
  imageSize = frame.total()*frame.elemSize();
  try
  {
    TCPSocket sock(server, clientPort);
    std::cout << "IMG_HEIGHT * IMG_WIDTH * 3: " << IMG_HEIGHT * IMG_WIDTH * 3
      << std::endl;
    std::cout << "imageSize: " << imageSize << std::endl;
    // std::cout << "frame.data: " << frame.data << std::endl;
    std::ofstream myfile;
    myfile.open ("client.txt");
    myfile << frame.data;
    myfile.close();
    sock.send(frame.data, imageSize);

    std::cout << "frame.data type: " << typeid(frame.data).name() << std::endl;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  
  return 0;
}