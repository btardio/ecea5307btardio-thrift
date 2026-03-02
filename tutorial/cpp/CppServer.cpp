/* 
 * ECEA 5307
 * Brandon Tardio
 * https://github.com/btardio/final-project-btardio
 * 
 * This file is a thrift server that runs on the RPI
 * it is compiled using the following command in docker:
 * 
 * git pull && git submodule foreach 'git pull' && sudo docker build -t btardio/ecea5307btardio:latest . && sudo docker push btardio/ecea5307btardio:latest
 * 
 * This command is on AWS EC-2 Arm architecture machine.
 * 
 * This produces a dockerhub image that the RPI pulls and runs.
 * 
 * docker container stop server || true && docker system prune -a -f && docker container run -it --network host --name server --privileged -v /sys/:/sys/ btardio/ecea5307btardio
 * 
 * This demonstrates STL and thrift.
 */


#include <CL/cl.h>
#define VECTOR_SIZE 1024

#include <thrift/concurrency/ThreadManager.h>
#include <thrift/concurrency/ThreadFactory.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/TToString.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "../gen-cpp/rgbatransform.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

using namespace tutorial;
using namespace shared;

class RgbaHandler : public rgbatransformIf {
public:
  RgbaHandler();
  
  void ehlo() override { cout << "pong()" << '\n'; }
    
  void doMosul(std::vector<long int>&, const std::vector<long int>&) override;
  
  void doMosulA(std::vector<rgbastruct>&, const std::vector<rgbastruct>&, const int width, const int height) override;
  
};

RgbaHandler::RgbaHandler() {

}


void RgbaHandler::doMosulA(std::vector<rgbastruct>& transformedImage, const std::vector<rgbastruct>& loadedImage, const int width, const int height) {

	for (size_t i = 0; i < loadedImage.size(); ++i) {
		
		rgbastruct transformed_pixel_rgbastruct;
		

		float luminance = (0.299f * (float)(loadedImage[i].r / 255.0f)) + (0.587f * (float)(loadedImage[i].g / 255.0f)) + (0.114f * (float)(loadedImage[i].b / 255.0f));        
		transformed_pixel_rgbastruct.r = (char)(luminance * 255.0f); // loadedImage[i].r; // (unsigned char)
		transformed_pixel_rgbastruct.g = (char)(luminance * 255.0f); // loadedImage[i].g;
		transformed_pixel_rgbastruct.b = (char)(luminance * 255.0f); // loadedImage[i].b; //(unsigned char)
		transformed_pixel_rgbastruct.a = (char)loadedImage[i].a;

//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(loadedImage[i].r)) << static_cast<unsigned int>(static_cast<unsigned char>(transformed_pixel_rgbastruct.r)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(loadedImage[i].g)) << static_cast<unsigned int>(static_cast<unsigned char>(transformed_pixel_rgbastruct.g)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(loadedImage[i].b)) << static_cast<unsigned int>(static_cast<unsigned char>(transformed_pixel_rgbastruct.b)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(loadedImage[i].a)) << static_cast<unsigned int>(static_cast<unsigned char>(transformed_pixel_rgbastruct.a)) << std::endl;

		transformedImage.push_back(transformed_pixel_rgbastruct);
	}
}

void RgbaHandler::doMosul(std::vector<long int>& a, const std::vector<long int>& b) {
	
	
}



int main() {

  // This server only allows one connection at a time, but spawns no threads
  TSimpleServer server(
    std::make_shared<rgbatransformProcessor>(std::make_shared<RgbaHandler>()),
    std::make_shared<TServerSocket>(9090),
    std::make_shared<TBufferedTransportFactory>(),
    std::make_shared<TBinaryProtocolFactory>());


  cout << "Starting the server..." << '\n';
  server.serve();
  cout << "Done." << '\n';
  return 0;
}

