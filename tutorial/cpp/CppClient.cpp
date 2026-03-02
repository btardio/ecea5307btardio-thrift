/* 
 * ECEA 5307
 * Brandon Tardio
 * https://github.com/btardio/final-project-btardio
 * 
 * This file is a thrift client that runs in docker
 * it is compiled using the following command in docker:
 * 
 * docker build -f ./Dockerfile_client -t thriftclient . && docker container run  -it thriftclient | python3 rgba_to_image.py
 * 
 * If the RPI server is running at address 192.168.1.100 it will produce 
 * \x00 char output.
 * 
 * The container run command uses the python script and the untitled.png
 * 
 * The file is read from stdin.
 */


#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include <arpa/inet.h>

#include "../gen-cpp/Calculator.h"

#include "../gen-cpp/rgbatransform.h"

#include <png.h>

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace tutorial;
using namespace shared;

std::string filename_create(){
	// 1. Seed the random number generator once per program run
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 2. Generate a random number.
    long long randomNum = rand(); // May be small

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

// Custom read function for std::istream
void png_read_istream(png_structp png_ptr, png_bytep data, png_size_t length) {
    auto* stream = reinterpret_cast<std::istream*>(png_get_io_ptr(png_ptr));
    stream->read(reinterpret_cast<char*>(data), length);
    //if (!*stream) png_error(png_ptr"Read error");
}

std::vector<rgbastruct> readPng(std::istream& inputFile) {
    std::vector<rgbastruct> pixels;

    // 1. Initialize libpng
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return pixels;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return pixels;
    }

//    if (setjmp(png_jmpbuf(png_ptr))) {
//        png_destroy_read_struct(&png_ptr&info_ptr, nullptr);
//        return pixels;
//    }

    // 2. Set up custom I/O
    png_set_read_fn(png_ptr, &inputFile, png_read_istream);

    // 3. Read info and set transforms to force 8-bit RGBA
    png_read_info(png_ptr, info_ptr);
    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (!(color_type & PNG_COLOR_MASK_ALPHA)) png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);
    pixels.resize(width * height);

    // 4. Read image data
    std::vector<png_bytep> row_pointers(height);
    for (png_uint_32 y = 0; y < height; y++) {
        row_pointers[y] = reinterpret_cast<png_bytep>(&pixels[y * width]);
    }

    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, nullptr);
    png_destroy_read_struct(&png_ptr,&info_ptr, nullptr);

    return pixels;
}

int main() { // int argc, char* argv[]) {
	int argc;
	char* argv[3];
	
	if (!std::cin) {
		std::cout << "stdin error\n";
		return -1;
	}
    
    std::string filename = "/images/untitled.png";

    std::ifstream fileStream(filename);

    if (!fileStream.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return 1;
    }

    std::shared_ptr<TTransport> socket(new TSocket("192.168.1.100", 9090));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

    rgbatransformClient rgbaclient(protocol);

    try {
		transport->open();
		std::vector<long int> my_vector = {1000000000L, 2000000000L, 3000000000L};
		cout << "ping rgba" << '\n';
	
		rgbaclient.ehlo();
    
		std::vector<rgbastruct> outvector;
		
		
		uint32_t a_width = 100;
		uint32_t a_height = 100;

		std::vector<rgbastruct> a_loadedImage = readPng(std::cin);

		
		for (std::vector<rgbastruct>::const_iterator it = a_loadedImage.begin(); it != a_loadedImage.end(); ++it) {
			// Access members, e.g., it->r
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
		}

		cout << "sane\n";
		
		rgbaclient.doMosulA(outvector, a_loadedImage, a_width, a_height);
		
		for (std::vector<rgbastruct>::const_iterator it = outvector.begin(); it != outvector.end(); ++it) {
			// Access members, e.g., it->r
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->r)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->g)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->b)) << ",";
			cout << "item: " << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(it->a)) << std::endl;
		}
		
		
		//writeRGBA("/out_transformed_image.rgba", a_width, a_height, outvector);
		

		try {
		  cout << "Whoa? We can divide by zero!" << '\n';
		} catch (InvalidOperation& io) {
		  cout << "InvalidOperation: " << io.why << '\n';

		}

		transport->close();
    } catch (TException& tx) {
      cout << "ERROR: " << tx.what() << '\n';
    }
}


