#ifndef ICEIMAGECOMPRESSION_H
#define ICEIMAGECOMPRESSION_H

#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include <stdio.h>
#include <jpeglib.h>
#include "Ice/ImageIce.ice.H"





class IceImageCompressor
{
  public:
    IceImageCompressor();

    std::vector<unsigned char> CompressImage(Image<PixRGB<byte> > input);

  private:

    //Structure to hold all of our jpeg compression parameters
    struct jpeg_compress_struct cinfo;

    //Structure to handle all of our errors
    struct jpeg_error_mgr jerr;

    GLOBAL(void) InitImageIceDest(std::vector<unsigned char>* destination_buffer);



};

#endif

