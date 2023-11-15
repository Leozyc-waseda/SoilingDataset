#ifndef ICEIMAGEDECOMPRESSION_H
#define ICEIMAGEDECOMPRESSION_H

#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include <stdio.h>
#include <jpeglib.h>
#include "Ice/ImageIce.ice.H"

class IceImageDecompressor
{
  public:
    IceImageDecompressor();

    Image<PixRGB<byte> > DecompressImage(std::vector<unsigned char> &img_buffer);

  private:

    //Structure to hold all of our jpeg compression parameters
    struct jpeg_decompress_struct cinfo;

    //Structure to handle all of our errors
    struct jpeg_error_mgr jerr;

    GLOBAL(void) InitImageIceSource(std::vector<unsigned char>* source_buffer);



};

#endif
