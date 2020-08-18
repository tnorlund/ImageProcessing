/**
 *   This accepts a buffered image from a client and writes the image data.
 */

#include <iostream>
#include <exception>
#include <string>
#include <regex>
#include <thread>
#include <vector>

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
 *   Handle the parameters given in the command line.
 * 
 *   @param port the port used to connect to the server
 *   @returns whether the parameters passed are proper or not
 */
bool processCommandLine(int argc, char** argv,
                          std::string& port) {
  try {
    boost::program_options::options_description description("Allowed options");
    description.add_options()
      ("help,h", "Help screen")
      ("port,p", boost::program_options::value(&port)->required(), 
        "the port used to connect to the client");
    
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv)
            .options(description).run (), 
        vm
    );

    if (vm.count("help")) {
      std::cout << description << std::endl;
      return false;
    }

    boost::program_options::notify(vm);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
  } catch(...) {
    std::cerr << "Unknown error!" << std::endl;
    return false;
  }
  return true;
}

/**
 *   Handle the client by receiveing the packet and sending it back.
 * 
 *   @param clientSocket the open socket used to accept the packet
 */
void HandleTCPClient(TCPSocket *clientSocket) {
  char messageBuffer[IMG_HEIGHT * IMG_WIDTH * 3];
  int messageLength;
  
  cv::Mat img = cv::Mat::zeros( IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
  int  imgSize = img.total()*img.elemSize();
  uchar sockData[imgSize];
  int bytes = 0;
  int ptr=0;
    
  std::cout << "Handling client ";
  try {
    std::cout << clientSocket->getForeignAddress() << ":";
  } catch (SocketException &e) {
    std::cerr << "Unable to get foreing address" << std::endl;
  }

  try {
    std::cout << clientSocket->getForeignPort();
  } catch(SocketException &e) {
    std::cerr << "Unable to get foreign port" << std::endl;
  }
  std::cout << " with thread " << std::this_thread::get_id() << std::endl;

  // for ()
  std::cout << "IMG_HEIGHT * IMG_WIDTH * 3: " << IMG_HEIGHT * IMG_WIDTH * 3 << std::endl;
  std::cout << "(IMG_HEIGHT * IMG_WIDTH * 3) / RCVBUFFERSIZE: " << (IMG_HEIGHT * IMG_WIDTH * 3) / RCVBUFFERSIZE << std::endl;

  for (int i = 0; i < imgSize; i += bytes) {
    bytes = clientSocket->recv(sockData+i, IMG_HEIGHT * IMG_WIDTH * 3);
    std::cout << bytes << std::endl;
  }

  std::cout << "sockData[0] type: " << typeid(sockData[0]).name() << std::endl;

  // messageLength = clientSocket->recv(sockData, IMG_HEIGHT * IMG_WIDTH * 3);
  std::ofstream myfile;
  myfile.open ("server.txt");
  myfile << sockData;
  myfile.close();



  for (size_t i = 0; i < IMG_HEIGHT; i++) {
    for (size_t j = 0; j < IMG_WIDTH; j++) {
      img.at<cv::Vec3b>(i,j) = cv::Vec3b(sockData[ptr+0],sockData[ptr+1],sockData[ptr+2]);
      ptr = ptr + 3;
    }
  }
  cv::imwrite("Tyler_Norlund_server.png", img);

  // Luckily, the destructor of the socket closes everything.
}

void *ThreadMain(void *clientSocket) {
  HandleTCPClient((TCPSocket *) clientSocket);
  delete (TCPSocket *) clientSocket;
  return NULL;
}

int main(int argc, char** argv) {
  bool result;
  std::string port;
  unsigned short serverPort;
  // Handle the arguments passed via command line
  result = processCommandLine(argc, argv, port);
  if (!result)
    return 1;
  
  serverPort = std::stoi(port);

  try {
    TCPServerSocket serverSocket(serverPort);
    while (true) {
      // Memory must be allocated in order to accept the client's argument.
      TCPSocket *clientSocket = serverSocket.accept();
      std::thread thread(ThreadMain, (void *) clientSocket);
      thread.detach();
    }
  } catch (SocketException &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
  return 0;
}