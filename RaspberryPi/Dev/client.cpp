#include <iostream>
#include <exception>
#include <string>
#include <regex>

#include "MPU6050.h"
#include "Socket.h"
#include <boost/program_options.hpp>

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
 *   @param message the message sent to the server
 *   @param port the port used to connect to the server
 *   @returns whether the parameters passed are proper or not
 */
bool processCommandLine(int argc, char** argv,
                          std::string& server,
                          std::string& message,
                          std::string& port) {
  try {
    boost::program_options::options_description description("Allowed options");
    description.add_options()
      ("help,h", "Help screen")
      ("server,s", boost::program_options::value(&server)->required(), 
        "the server IP address")
      ("message,m", boost::program_options::value(&message)->required(), 
        "the message sent to the server")
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

    boost::program_options::notify( vm );
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
  std::string server, message, port;
  int messageLength;
  unsigned short serverPort;
  char echoBuffer[RCVBUFFERSIZE + 1];
  /**
   *   The character buffer used to store what is being sent to the server.
   */
  char * messageBuffer;
  int bytesReceived = 0;
  int totalBytesReceived = 0;

  // Handle the arguments passed via command line
  result = processCommandLine(argc, argv, server, message, port);
  if (!result)
    return 1;

  // Check to see if the camera is connected an accesible.
  checkCameraConfiguration();

  // The TCP socket uses a buffer to stream data. In order to use the buffer, 
  // the string passed as a command line argumenet must be discretized as an
  // array of characters.
  messageBuffer = (char *)malloc((message.size() + 1) * sizeof(char));
  message.copy(messageBuffer, message.size() + 1);
  messageBuffer[message.size() + 1] = '\0';

  messageLength = message.length();
  serverPort = std::stoi(port);

  try {
    // Initialize a connection to the server
    TCPSocket sock(server, serverPort);
    // Send the message to the server
    sock.send(messageBuffer, messageLength);
    // Receive the same message back. Since the message length is known, the 
    // buffer can be used.
    while (totalBytesReceived < messageLength) {
      // When the bytesRecieved is negative, the packet received is no longer 
      // a part of the buffer. At this point the packet is no longer what the
      // server is sending back.
      if ((bytesReceived = (sock.recv(echoBuffer, RCVBUFFERSIZE))) <= 0) {
        std::cerr << "Unable to read";
        exit(1);
      }
      totalBytesReceived += bytesReceived;
      echoBuffer[bytesReceived] = '\0';
      std::cout << echoBuffer;
    }
    std::cout << std::endl;  

  } catch(SocketException &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  std::cout << "It's working!" << std::endl;

  return 0;
}