/*
 * Licensed to the Apache Software Foundation (ASF) under one
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


#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "../gen-cpp/Calculator.h"

#include "../gen-cpp/rgbatransform.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace tutorial;
using namespace shared;
//using namespace rgbatransform;



//struct Pixel {
//    bool r, g, b, a;
//};

// Write an 8x8x8x8 RGBA file
void writeRGBA(const std::string& filename, int width, int height, const std::vector<rgbastruct>& pixels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    // Write raw pixel data directly
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(rgbastruct));
    file.close();
}

// Read an 8x8x8x8 RGBA file
std::vector<rgbastruct> readRGBA(const std::string& filename, int width, int height) {
    
    cout << filename << "_a_filename\n";
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return {};
    }
    std::vector<rgbastruct> pixels(width * height);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(rgbastruct));
    file.close();
    return pixels;
}




int main(int argc, char* argv[]) {
	
	

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <string> <width> <height>" << std::endl;
        return 1;
    }

    std::string a_filename = argv[1]; // Get string
    int a_width = std::stoi(argv[2]); // Convert width
    int a_height = std::stoi(argv[3]); // Convert height

    std::cout << "Name: " << a_filename << ", Width: " << a_width << ", Height: " << a_height << std::endl;

	
	
	
	
  std::shared_ptr<TTransport> socket(new TSocket("192.168.1.100", 9090));
  std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  //CalculatorClient client(protocol);
  rgbatransformClient rgbaclient(protocol);
  
  //rgbaclient = rgbatransformClient;

  try {
    transport->open();
    std::vector<long int> my_vector = {1000000000L, 2000000000L, 3000000000L};
	cout << "ping rgba" << '\n';
	//rgbaclient.doMosul(my_vector, my_vector);
	
    rgbaclient.ehlo();
    
    
    
    std::vector<rgbastruct> outvector;
    
    
    
    int width = 8, height = 8;
    
    

	
    std::vector<rgbastruct> image(width * height); //(width * height, {0xFF, 0x00, 0x00, 0xFF}); // Red opaque pixels


	for (int i = 0; i < width * height; ++i) {
        image[i].r = (char)0xFF;
        image[i].g = (char)0x00;
        image[i].b = (char)0x00;
        image[i].a = (char)0xFF;
    }
	



    // Write
    writeRGBA("test.rgba", width, height, image);

	std::vector<rgbastruct> a_loadedImage = readRGBA("input.rgba", width, height);

    // Read
    std::vector<rgbastruct> loadedImage = readRGBA("test.rgba", width, height);

    
    for (std::vector<rgbastruct>::const_iterator it = loadedImage.begin(); it != loadedImage.end(); ++it) {
		// Access members, e.g., it->r
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
//		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
	}

    
    cout << a_width << "a_width\n";
    cout << a_height << "a_heigth\n";
    cout << a_filename << "a_filename\n";
    rgbaclient.doMosulA(outvector, loadedImage, width, height);
    
    
    
    
    
    
	for (std::vector<rgbastruct>::const_iterator it = outvector.begin(); it != outvector.end(); ++it) {
		// Access members, e.g., it->r
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
		cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
	}
    
    
    //cout << "ping()" << '\n';

    //cout << "1 + 1 = " << client.add(1, 1) << '\n';

    Work work;
    work.op = Operation::DIVIDE;
    work.num1 = 1;
    work.num2 = 0;

    try {
      //client.calculate(1, work);
      cout << "Whoa? We can divide by zero!" << '\n';
    } catch (InvalidOperation& io) {
      cout << "InvalidOperation: " << io.why << '\n';
      // or using generated operator<<: cout << io << '\n';
      // or by using std::exception native method what(): cout << io.what() << '\n';
    }

    work.op = Operation::SUBTRACT;
    work.num1 = 15;
    work.num2 = 10;
    //int32_t diff = client.calculate(1, work);
    //cout << "15 - 10 = " << diff << '\n';

    // Note that C++ uses return by reference for complex types to avoid
    // costly copy construction
    SharedStruct ss;
    ///client.getStruct(ss, 1);
    cout << "Received log: " << ss << '\n';

    transport->close();
  } catch (TException& tx) {
    cout << "ERROR: " << tx.what() << '\n';
  }
}
