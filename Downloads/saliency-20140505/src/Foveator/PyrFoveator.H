/*!@file Foveator/PyrFoveator.H Foveator class that builds dyadic pyramid */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/PyrFoveator.H $
// $Id: PyrFoveator.H 4663 2005-06-23 17:47:28Z rjpeters $

#ifndef PYRFOVEATOR_H_DEFINED
#define PYRFOVEATOR_H_DEFINED

#include "Foveator/Foveator.H"
#include "Image/PyrBuilder.H"

//! A class that does space-variant processing with dyadic pyramid
/*! A Gaussian pyramid is constructed from the original image, each level
with resolution halved, image size 1/4 of previous level.  After building the
pyramid, a foveated image is reassembled by starting with low-resolution
image expanded to full-size, then superimposing smaller rectangles of higher
resolution.

Testing has shown that PyrFoveator is somewhat faster than BlurFoveator and
can perform similar blurring, but PyrFoveator creates noticeable straight-line
boundaries between rectangles of differing resolution.
*/

class PyrFoveator : public Foveator
{
private:
  GaussianPyrBuilder< PixRGB< byte > > builder;
  Dims baseRect;

public:
  //! Construct a PyrFoveator for img
  /*! This constructor will create a PyrFoveator that handles img and all
    other images of the same dimensions.
    @param img image to be foveated
    @param filterSize size of blur filter to be applied
  */
  PyrFoveator( const Image< PixRGB<byte> >& img, int filterSize );
  //! Construct a PyrFoveator from another
  PyrFoveator( const PyrFoveator& pf );
  //! Construct a PyrFoveator equal to another
  PyrFoveator& operator=( const PyrFoveator& pf );
  //! Destructor
  ~PyrFoveator();

  //! Set dimensions of smallest rectangle with highest resolution
  void setBaseRect( Dims &d );
  //! Set dimensions of smallest rectangle with highest resolution
  void setBaseRect( int x, int y );

  //! Perform dyadic pyramid foveation and return resulting image
  Image< PixRGB<byte> > foveate( void );
  //! Perform one dyadic pyramid foveation without a PyrFoveator
  /*! This function is for foveating ONE image with a call such as
    PyrFoveator::foveate().  It is not optimized for foveating multiple
    images.
    @param img image to be foveated
    @param filterSize size of blur filter to be applied
    @param baseRectWidth width of base rectangle
    @param baseRectHeight height of base rectangle
    @param x horizontal coordinate of origin
    @param y vertical coordinate of origin
  */
  static Image< PixRGB<byte> > foveate( const Image< PixRGB<byte> >& img,
                                      int filterSize, int baseRectWidth,
                                      int baseRectHeight, int x, int y );
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
