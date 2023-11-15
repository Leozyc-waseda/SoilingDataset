#ifndef ICEIMAGECOMPRESSION_C
#define ICEIMAGECOMPRESSION_C

#include "Ice/IceImageCompressor.H"
#include <vector>

//The output buffer size... I believe this should be as large as possible
//so that empty_output_buffer will never need to be called as an intermediary
//step, and we can just rely on term_destination to write all of our data
#define OUTPUT_BUF_SIZE  100000

namespace IIC_NS
{
  typedef struct {
    //The public fields common to destination managers
    struct jpeg_destination_mgr pub;

    //The input buffer pointer
    JOCTET* in_buffer;

    //The destination queue of
    std::vector<unsigned char>* out_buffer;

  } buff_dest_mgr;

  ////////////////////////////////////////////////////////////////
  // Initialize destination --- called by jpeg_start_compress
  // before any data is actually written.
  ////////////////////////////////////////////////////////////////
  METHODDEF(void) init_destination(j_compress_ptr cinfo)
  {
    //Get the pointer to our cinfo's destination manager
    buff_dest_mgr* dest = (buff_dest_mgr*) cinfo->dest;

    //Allocate the input buffer --- it will be released when done with image
    dest->in_buffer = (JOCTET *)
      (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
          OUTPUT_BUF_SIZE * sizeof(JOCTET));

    //Reset the input buffer
    dest->pub.next_output_byte = dest->in_buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
  }

  ////////////////////////////////////////////////////////////////
  // Empty the output buffer --- called whenever buffer fills up.
  ////////////////////////////////////////////////////////////////
  METHODDEF(boolean) empty_output_buffer (j_compress_ptr cinfo)
  {
    //Get the pointer to our cinfo's destination manager
    buff_dest_mgr* dest = (buff_dest_mgr*) cinfo->dest;

    //Copy the rest of the input buffer onto the end of our output buffer
    dest->out_buffer->insert(dest->out_buffer->end(),
        dest->in_buffer,
        dest->in_buffer+OUTPUT_BUF_SIZE);

    //Reset the input buffer
    dest->pub.next_output_byte = dest->in_buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

    return TRUE;
  }

  ////////////////////////////////////////////////////////////////
  // Terminate destination --- called by jpeg_finish_compress
  // after all data has been written.  Usually needs to flush buffer.
  ////////////////////////////////////////////////////////////////
  METHODDEF(void) term_destination (j_compress_ptr cinfo)
  {
    //Get the pointer to our cinfo's destination manager
    buff_dest_mgr* dest = (buff_dest_mgr*) cinfo->dest;

    //Calculate the number of bytes left to be written
    size_t datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;

    //Copy the rest of the input buffer onto the end of our output buffer
    dest->out_buffer->insert(dest->out_buffer->end(),
        dest->in_buffer,
        dest->in_buffer+datacount);
  }
}


IceImageCompressor::IceImageCompressor()
{
  //Initialize our jpeg error handler to the default
  //(spit out non-fatal error messages on cerr, and
  // exit() on fatal errors)
  cinfo.err = jpeg_std_error(&jerr);

  //Create our jpeg compression object
  jpeg_create_compress(&cinfo);
}

std::vector<unsigned char> IceImageCompressor::CompressImage(Image<PixRGB<byte> > input)
{
  std::vector<unsigned char> jpeg_buffer;

  //Initialize our data destination
  InitImageIceDest(&jpeg_buffer);

  //Set our image and compression parameters
  cinfo.image_width   = input.getWidth();
  cinfo.image_height  = input.getHeight();
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);

  //Begin the compression
  jpeg_start_compress(&cinfo, TRUE);

  //Get a pointer to the start of the image's raw data array. This assumes that
  //all of the image data is layed out as R G B R G B, with each element as a
  //byte in contiguous memory. This should be a valid assumption for any
  //Image<PixRGB<byte> >
  JSAMPLE* p_start = reinterpret_cast<JSAMPLE*>(input.getArrayPtr());

  //Pass a pointer to each row of the image to the jpeg compressor.  It would
  //be nice to do this all in one shot, but libjpeg segfaults when I try.
  for(int row_idx=0; row_idx<input.getHeight(); row_idx++)
  {
    JSAMPLE* p = (p_start + (row_idx * input.getWidth()*3) );
    jpeg_write_scanlines(&cinfo, &p, 1);
  }

  //Clean up the compression, and finish writing all bytes to the output buffer
  jpeg_finish_compress(&cinfo);

  return jpeg_buffer;
}


void IceImageCompressor::InitImageIceDest(std::vector<unsigned char>* output_buffer)
{
  IIC_NS::buff_dest_mgr* dest;

  if(cinfo.dest == NULL)
  {
    //Allocate some memory space for our destination manager.
    cinfo.dest = (struct jpeg_destination_mgr *)
      (*(cinfo.mem->alloc_small)) ((j_common_ptr) &cinfo, JPOOL_PERMANENT, sizeof(IIC_NS::buff_dest_mgr));
  }

  //Initialize our destination manager by filling in all of the appropriate
  //function pointers, and assigning the output buffer.
  dest = (IIC_NS::buff_dest_mgr*) cinfo.dest;
  dest->pub.init_destination    = IIC_NS::init_destination;
  dest->pub.empty_output_buffer = IIC_NS::empty_output_buffer;
  dest->pub.term_destination    = IIC_NS::term_destination;
  dest->out_buffer              = output_buffer;
}

#endif

