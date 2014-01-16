
#include "SimpleOpt.h"
#include "JPEGPyramid.h"
#include "JPEGImage.h" 
#include "Patchwork.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace FFLD;
using namespace std;

//TODO: have a pyramid stitch class?


//e.g. file = ../../images_640x480/carsgraz_001.image.jpg
string parse_base_filename(string file){
    size_t lastSlash = file.find_last_of("/\\");
    size_t lastDot = file.find_last_of('.');

    string base_filename = file.substr(lastSlash, lastDot-lastSlash); // /carsgraz_001.image
    return base_filename;
}



//int main(int argc, char * argv[])
Patchwork stitch_pyramid(string file, int padding=8, int interval=10)
{
    string base_filename = parse_base_filename(file);
    printf("    base_filename = %s \n", base_filename.c_str());
 
	JPEGImage image(file);
    if (image.empty()) {
        cerr << "\nInvalid image " << file << endl;
    }

    //image = image.resize(image.width()*4, image.height()*4); //UPSAMPLE so that Caffe's 16x downsampling looks like 4x downsampling
    image = image.resize(image.width()*2, image.height()*2); //UPSAMPLE so that Caffe's 16x downsampling looks like 8x downsampling

  // Compute the downsample+stitch
    JPEGPyramid pyramid(image, padding, padding, interval); //DOWNSAMPLE with (padx == pady == padding)
    if (pyramid.empty()) {
        cerr << "\nInvalid image " << file << endl;
    }
    
    int planeWidth = (pyramid.levels()[0].width() + 15) & ~15; //TODO: don't subtract padx, pady? 
    int planeHeight = (pyramid.levels()[0].height() + 15) & ~15; 
    planeWidth = max(planeWidth, planeHeight);  //SQUARE planes for Caffe convnet
    planeHeight = max(planeWidth, planeHeight);

    Patchwork::Init(planeHeight, planeWidth); 
    const Patchwork patchwork(pyramid); //STITCH

    return patchwork;
}

//@param convnet_subsampling_ratio = difference between input image dim and convnet feature dim
//         e.g. if input img is 200x200 and conv5 is 25x25 ... 200/25=8 -> 8x downsampling in convnet
JPEGPyramid unstitch_pyramid(Patchwork image_patchwork, float* convnet_planes, int convnet_subsampling_ratio){

    JPEGPyramid pyra; //stub

    return pyra;
}


