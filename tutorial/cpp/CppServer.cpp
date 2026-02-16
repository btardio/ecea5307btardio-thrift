/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
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

#include "../gen-cpp/Calculator.h"

#include "../gen-cpp/rgbatransform.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

using namespace tutorial;
using namespace shared;

//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(                         \n"
"                  __global const uchar *A, \n"
"                  __global const uchar *B, \n"
"                  __global const uchar *C, \n"
"                  __global uchar *OUT)     \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    OUT[index] = 0xAD;                     \n"
"}                                          \n";

//convert_uchar(convert_float(A[index] + B[index] + C[index]) / 255.0f); \n"

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
	cout << "aaaping()" << '\n';
	
	
    for (std::vector<rgbastruct>::const_iterator it = loadedImage.begin(); it != loadedImage.end(); ++it) {
		// Access members, e.g., it->r
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
	}

	for (size_t i = 0; i < loadedImage.size(); ++i) {
		
		rgbastruct transformed_pixel_rgbastruct;
		
		float luminance = 0.299f * (float)loadedImage[i].r + 0.587f * (float)loadedImage[i].g + 0.114f * (float)loadedImage[i].b;        
		transformed_pixel_rgbastruct.r = (unsigned char)(luminance * 255.0f);
		transformed_pixel_rgbastruct.g = (unsigned char)(luminance * 255.0f);
		transformed_pixel_rgbastruct.b = (unsigned char)(luminance * 255.0f);
		transformed_pixel_rgbastruct.a = 0xFF;
		transformedImage.push_back(transformed_pixel_rgbastruct);
	}
	

	cout << "H: " << transformedImage.size() << std::endl;


    for (std::vector<rgbastruct>::const_iterator it = transformedImage.begin(); it != transformedImage.end(); ++it) {
		// Access members, e.g., it->r
		cout << "transformedImage: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
		cout << "transformedImage: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
		cout << "transformedImage: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
		cout << "transformedImage: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
	}

	cout << std::dec << std::setw(0) << std::setfill(' ');
	cout << "loadedImage.size(): " << loadedImage.size() << std::endl;
	cout << "transformedImage.size(): " << transformedImage.size() << std::endl;

	// Finally release all OpenCL allocated objects and host buffers.

	
	
}

void RgbaHandler::doMosul(std::vector<long int>& a, const std::vector<long int>& b) {
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}


class CalculatorHandler : public CalculatorIf {
public:
  CalculatorHandler() = default;

  void ping() override { cout << "aaaping()" << '\n'; }

  int32_t add(const int32_t n1, const int32_t n2) override {
    cout << "add(" << n1 << ", " << n2 << ")" << '\n';
    return n1 + n2;
  }

  int32_t calculate(const int32_t logid, const Work& work) override {



	    int i;
  // Allocate space for vectors A, B and C
  float alpha = 2.0;
  float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  for(i = 0; i < VECTOR_SIZE; i++)
  {
    A[i] = i;
    B[i] = VECTOR_SIZE - i;
    C[i] = 0;
  }

  // Get platform and device information
  cl_platform_id * platforms = NULL;
  cl_uint     num_platforms;
  //Set up the Platform
  cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)
  malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

  //Get the devices list and choose the device you want to run on
  cl_device_id     *device_list = NULL;
  cl_uint           num_devices;

  clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_CPU, 0,NULL, &num_devices);
  device_list = (cl_device_id *)
  malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_CPU, num_devices, device_list, NULL);

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, VECTOR_SIZE * sizeof(float), NULL, &clStatus);

  // Copy the Buffer A and B to the device
  clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);

  // Build the program
  clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

      alpha = (float)i;
      if (i > 65535) {
	      i = 0;
      }
      // get a new item from a queue
      //
      // Set the arguments of the kernel
      clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
      clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
      clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
      clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

      // Execute the OpenCL kernel on the list
      size_t global_size = VECTOR_SIZE; // Process the entire lists
      size_t local_size = 64;           // Process one item at a time
      clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

      // Read the cl memory C_clmem on device to the host variable C
      clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);

      // Clean up and wait for all the comands to complete.
      clStatus = clFlush(command_queue);
      clStatus = clFinish(command_queue);

  for(i = 0; i < VECTOR_SIZE; i++)
    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);






  // Finally release all OpenCL allocated objects and host buffers.
  clStatus = clReleaseKernel(kernel);
  clStatus = clReleaseProgram(program);
  clStatus = clReleaseMemObject(A_clmem);
  clStatus = clReleaseMemObject(B_clmem);
  clStatus = clReleaseMemObject(C_clmem);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);
  free(A);
  free(B);
  free(C);
  free(platforms);
  free(device_list);

      

    cout << "calculate(" << logid << ", " << work << ")" << '\n';
    int32_t val;

    switch (work.op) {
    case Operation::ADD:
      val = work.num1 + work.num2;
      break;
    case Operation::SUBTRACT:
      val = work.num1 - work.num2;
      break;
    case Operation::MULTIPLY:
      val = work.num1 * work.num2;
      break;
    case Operation::DIVIDE:
      if (work.num2 == 0) {
        InvalidOperation io;
        io.whatOp = work.op;
        io.why = "Cannot divide by 0";
        throw io;
      }
      val = work.num1 / work.num2;
      break;
    default:
      InvalidOperation io;
      io.whatOp = work.op;
      io.why = "Invalid Operation";
      throw io;
    }

    SharedStruct ss;
    ss.key = logid;
    ss.value = to_string(val);

    log[logid] = ss;

    return val;
  }

  void getStruct(SharedStruct& ret, const int32_t logid) override {
    cout << "getStruct(" << logid << ")" << '\n';
    ret = log[logid];
  }

  void zip() override { cout << "zip()" << '\n'; }

protected:
  map<int32_t, SharedStruct> log;
};

/*
  CalculatorIfFactory is code generated.
  CalculatorCloneFactory is useful for getting access to the server side of the
  transport.  It is also useful for making per-connection state.  Without this
  CloneFactory, all connections will end up sharing the same handler instance.
*/
class CalculatorCloneFactory : virtual public CalculatorIfFactory {
 public:
  ~CalculatorCloneFactory() override = default;
  CalculatorIf* getHandler(const ::apache::thrift::TConnectionInfo& connInfo) override
  {
    std::shared_ptr<TSocket> sock = std::dynamic_pointer_cast<TSocket>(connInfo.transport);
    cout << "Incoming connection\n";
    cout << "\tSocketInfo: "  << sock->getSocketInfo() << "\n";
    cout << "\tPeerHost: "    << sock->getPeerHost() << "\n";
    cout << "\tPeerAddress: " << sock->getPeerAddress() << "\n";
    cout << "\tPeerPort: "    << sock->getPeerPort() << "\n";
    return new CalculatorHandler;
  }
  void releaseHandler( ::shared::SharedServiceIf* handler) override {
    delete handler;
  }
};

//rgbatransformIfFactory
//class rgbatransformCloneFactory : virtual public rgbatransformIfFactory {
// public:
//  ~rgbatransformCloneFactory() override = default;
//  rgbatransformIf* getHandler(const ::apache::thrift::TConnectionInfo& connInfo) override
//  {
//    std::shared_ptr<TSocket> sock = std::dynamic_pointer_cast<TSocket>(connInfo.transport);
//    cout << "Incoming connection\n";
//    cout << "\tSocketInfo: "  << sock->getSocketInfo() << "\n";
//    cout << "\tPeerHost: "    << sock->getPeerHost() << "\n";
//    cout << "\tPeerAddress: " << sock->getPeerAddress() << "\n";
//    cout << "\tPeerPort: "    << sock->getPeerPort() << "\n";
//    return new RgbaHandler;
//  }
//  void releaseHandler( ::shared::SharedServiceIf* handler) override {
//    delete handler;
//  }
//};


int main() {
//  TThreadedServer server(
//    std::make_shared<CalculatorProcessorFactory>(std::make_shared<CalculatorCloneFactory>()),
//    std::make_shared<TServerSocket>(9090), //port
//    std::make_shared<TBufferedTransportFactory>(),
//    std::make_shared<TBinaryProtocolFactory>());

  
  // if you don't need per-connection state, do the following instead
  TThreadedServer server(
    std::make_shared<rgbatransformProcessor>(std::make_shared<RgbaHandler>()),
    std::make_shared<TServerSocket>(9090), //port
    std::make_shared<TBufferedTransportFactory>(),
    std::make_shared<TBinaryProtocolFactory>());
  

  
  // * Here are some alternate server types...
/*
  // This server only allows one connection at a time, but spawns no threads
  TSimpleServer server(
    std::make_shared<CalculatorProcessor>(std::make_shared<CalculatorHandler>()),
    std::make_shared<TServerSocket>(9090),
    std::make_shared<TBufferedTransportFactory>(),
    std::make_shared<TBinaryProtocolFactory>());

  const int workerCount = 4;

  std::shared_ptr<ThreadManager> threadManager =
    ThreadManager::newSimpleThreadManager(workerCount);
  threadManager->threadFactory(
    std::make_shared<ThreadFactory>());
  threadManager->start();

  // This server allows "workerCount" connection at a time, and reuses threads
  TThreadPoolServer server(
    std::make_shared<CalculatorProcessorFactory>(std::make_shared<CalculatorCloneFactory>()),
    std::make_shared<TServerSocket>(9090),
    std::make_shared<TBufferedTransportFactory>(),
    std::make_shared<TBinaryProtocolFactory>(),
    threadManager);
  */

  cout << "Starting the server..." << '\n';
  server.serve();
  cout << "Done." << '\n';
  return 0;
}

