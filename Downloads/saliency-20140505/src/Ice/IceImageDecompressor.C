#ifndef ICEIMAGEDECOMPRESSOR_C
#define ICEIMAGEDECOMPRESSOR_C

#include "Ice/IceImageDecompressor.H"
#include <vector>
#include <jpeglib.h>
#include <jerror.h>

#define INPUT_BUF_SIZE  4096LU

namespace IIDC_NS
{
  typedef struct
  {
    //The public fields common to source managers
    struct jpeg_source_mgr pub;

    //The compressed image source base pointer
    std::vector<unsigned char>* src_vec;
    int curr_src_pos;

    //The output buffer pointer
    JOCTET* buffer;

  } buff_src_mgr;


  ////////////////////////////////////////////////////////////////
  // Initialize the source -- called by jpeg_read_header before
  // any data is actually read
  ////////////////////////////////////////////////////////////////
  METHODDEF(void)
    init_source (j_decompress_ptr cinfo)
    {
      buff_src_mgr* src = (buff_src_mgr*) cinfo->src;
      src->curr_src_pos = 0;
    }

  ////////////////////////////////////////////////////////////////
  // Fill the input buffer -- called whenever bytes_in_buffer is 0
  ////////////////////////////////////////////////////////////////
  METHODDEF(boolean)
    fill_input_buffer (j_decompress_ptr cinfo)
    {

      buff_src_mgr* src = (buff_src_mgr*) cinfo->src;

      size_t nbytes;

      nbytes = std::min(INPUT_BUF_SIZE, src->src_vec->size() - src->curr_src_pos);
      unsigned char* src_base = &((*(src->src_vec))[0]);


      src->pub.next_input_byte =  src_base + src->curr_src_pos;
      src->pub.bytes_in_buffer = nbytes;
      src->curr_src_pos += nbytes;

      return TRUE;
    }

  ////////////////////////////////////////////////////////////////
  // Skip input data -- Skips num_bytes worth of data without
  // putting it into the buffer
  ////////////////////////////////////////////////////////////////
  METHODDEF(void)
    skip_input_data (j_decompress_ptr cinfo, long num_bytes)
    {
      buff_src_mgr* src = (buff_src_mgr*) cinfo->src;

      src->curr_src_pos += num_bytes;

      src->pub.next_input_byte += (size_t) num_bytes;
      src->pub.bytes_in_buffer -= (size_t) num_bytes;
    }

  ////////////////////////////////////////////////////////////////
  // Terminate source -- Nothing to do here
  ////////////////////////////////////////////////////////////////
  METHODDEF(void)
    term_source (j_decompress_ptr cinfo)
    {
      /* no work necessary here */
    }

}

// constuctor does nothing, maybe that needs FIXME
IceImageDecompressor::IceImageDecompressor()
{ }

////////////////////////////////////////////////////////////////
// Decompress an Image -- Pass this function an std::vector
// filled with data from a compressed jpeg image, and it will
// return to you an uncompressed Image< PixRGB<byte> >
////////////////////////////////////////////////////////////////
Image<PixRGB<byte> > IceImageDecompressor::DecompressImage(std::vector<unsigned char> &source_buffer)
{
  //Initialize our jpeg error handler to the default
  //(spit out non-fatal error messages on cerr, and
  // exit() on fatal errors)
  cinfo.err = jpeg_std_error(&jerr);

  //Create our jpeg decompression object
  jpeg_create_decompress(&cinfo);

  InitImageIceSource(&source_buffer);

  (void) jpeg_read_header(&cinfo, TRUE);
  (void) jpeg_start_decompress(&cinfo);


  assert(cinfo.output_components == 3);
  Image<PixRGB<byte> > outputImg(cinfo.output_width, cinfo.output_height, NO_INIT);
  JSAMPLE* img_ptr = reinterpret_cast<JSAMPLE*>(outputImg.getArrayPtr());

  int row_stride = cinfo.output_width * cinfo.output_components;

  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  while(cinfo.output_scanline < cinfo.output_height)
  {
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    std::memcpy(img_ptr, *buffer, row_stride);
    img_ptr+=row_stride;
  }


  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  return outputImg;
}


////////////////////////////////////////////////////////////////
// Initialize image source -- Just set up some variables and
// allocate some buffer space (if necessary) for the
// decompressor
////////////////////////////////////////////////////////////////
GLOBAL(void) IceImageDecompressor::InitImageIceSource(std::vector<unsigned char>* source_buffer)
{
  IIDC_NS::buff_src_mgr* src;

  //Allocate the source buffer
  if (cinfo.src == NULL)
  {        /* first time for this JPEG object? */

    cinfo.src = (struct jpeg_source_mgr *)
      (*cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_PERMANENT,
          sizeof(IIDC_NS::buff_src_mgr));

    src = (IIDC_NS::buff_src_mgr*) cinfo.src;

    src->buffer = (JOCTET *)
      (*cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_PERMANENT,
          INPUT_BUF_SIZE * sizeof(JOCTET));
  }


  src = (IIDC_NS::buff_src_mgr*) cinfo.src;
  src->pub.init_source = IIDC_NS::init_source;
  src->pub.fill_input_buffer = IIDC_NS::fill_input_buffer;
  src->pub.skip_input_data = IIDC_NS::skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
  src->pub.term_source = IIDC_NS::term_source;
  src->pub.bytes_in_buffer = 0; /* forces fill_input_buffer on first read */
  src->pub.next_input_byte = NULL; /* until buffer loaded */

  src->src_vec      = source_buffer;
  src->curr_src_pos = 0;

}
#endif
