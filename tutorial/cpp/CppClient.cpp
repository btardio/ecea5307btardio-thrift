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

std::string filename_create(){
	    // 1. Seed the random number generator once per program run
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 2. Generate a random number.
    // To ensure the number can potentially reach 10 digits, we need a large range.
    // We can use a modulo with a large number (e.g., 1 billion) or just use the full rand() value.
    // Note: RAND_MAX may be as small as 32767. For a 10-digit number, it is recommended to use
    // the C++11 <random> library for better random numbers and larger ranges.
    // If you must use `rand()`, be aware of its limitations.
    long long randomNum = rand(); // May be small

    // A more suitable range for a 10-digit number might involve combining rand() calls if RAND_MAX is small,
    // or using the C++11 random library. For this example, we proceed with the assumption of a sufficiently large number.

    // 3. Convert to a string with leading zeros using stringstream
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << randomNum;
    std::string randomStr = ss.str();

    // 4. Ensure the string is exactly length 10 (handle potential overflow if number is > 10 digits)
    if (randomStr.length() > 10) {
        randomStr = randomStr.substr(randomStr.length() - 10);
    }
    return randomStr;
}


std::vector<rgbastruct> readAndPrint4Bytes(const std::string& filename) {
    // Open the file in binary mode
    
    std::vector<rgbastruct> out;
    
    std::ifstream inputFile(filename, std::ios::in | std::ios::binary);

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return out;
    }

    // Buffer to hold 4 bytes
    unsigned char buffer[4];
    uint32_t value; // Use uint32_t to ensure a consistent 4-byte size across platforms

    // Read 4 bytes at a time until the end of the file
    while (inputFile.read(reinterpret_cast<char*>(buffer), sizeof(buffer))) {
        // Use static_cast<unsigned int>(static_cast<unsigned char>()) for each byte
        // and combine using bitwise shifts. This approach explicitly handles endianness if you define an order.
        // The following combines bytes in little-endian order (LSB first), common on x86 platforms.
        
        value = static_cast<uint32_t>(static_cast<unsigned int>(buffer[0])) |
                static_cast<uint32_t>(static_cast<unsigned int>(buffer[1])) << 8 |
                static_cast<uint32_t>(static_cast<unsigned int>(buffer[2])) << 16 |
                static_cast<uint32_t>(static_cast<unsigned int>(buffer[3])) << 24;

        // Print the value to the terminal (in hexadecimal for clarity)
        std::cout << "Read value (hex): 0x" << std::hex << std::setw(8) << std::setfill('0') << value << std::dec << std::endl;
        
        
		//std::vector<rgbastruct> image(width * height); //(width * height, {0xFF, 0x00, 0x00, 0xFF}); // Red opaque pixels


		rgbastruct transformed_pixel_rgbastruct;

		transformed_pixel_rgbastruct.r = (char)static_cast<uint32_t>(static_cast<unsigned int>(buffer[0]));
		transformed_pixel_rgbastruct.g = (char)static_cast<uint32_t>(static_cast<unsigned int>(buffer[1]));
		transformed_pixel_rgbastruct.b = (char)static_cast<uint32_t>(static_cast<unsigned int>(buffer[2]));
		transformed_pixel_rgbastruct.a = (char)static_cast<uint32_t>(static_cast<unsigned int>(buffer[3]));
		

		out.push_back(transformed_pixel_rgbastruct);
		
		//for (int i = 0; i < width * height; ++i) {
//			image[i].r = (char)0xFF;
//			image[i].g = (char)0x00;
//			image[i].b = (char)0x00;
//			image[i].a = (char)0xFF;
		//}
        
    }

    // Check if the loop terminated due to an error other than reaching the end of the file
    if (!inputFile.eof()) {
        std::cerr << "Error reading file!" << std::endl;
    }

    inputFile.close();
    return out;
}


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

		std::vector<rgbastruct> a_loadedImage = readAndPrint4Bytes(a_filename); //readRGBA("input.rgba", width, height);

		// Read
		//std::vector<rgbastruct> loadedImage = readRGBA("test.rgba", width, height);

		
		for (std::vector<rgbastruct>::const_iterator it = a_loadedImage.begin(); it != a_loadedImage.end(); ++it) {
			// Access members, e.g., it->r
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
		}

		
		cout << a_width << "a_width\n";
		cout << a_height << "a_heigth\n";
		cout << a_filename << "a_filename\n";
		rgbaclient.doMosulA(outvector, a_loadedImage, a_width, a_height);
		
		
		
		
		
		
		for (std::vector<rgbastruct>::const_iterator it = outvector.begin(); it != outvector.end(); ++it) {
			// Access members, e.g., it->r
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
		}
		
		
		writeRGBA("/images/transformed_image_" + filename_create() + ".rgba", a_width, a_height, outvector);
		
		
		
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
