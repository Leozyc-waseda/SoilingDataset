/*!@file Foveator/BlurFoveator.H Foveator class that performs progressive blurring */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/BlurFoveator.H $
// $Id: BlurFoveator.H 4663 2005-06-23 17:47:28Z rjpeters $

#ifndef BLURFOVEATOR_H_DEFINED
#define BLURFOVEATOR_H_DEFINED

#include "Foveator/Foveator.H"

//! A class that does space-variant processing with progressive blurring.
/*! A Gaussian blur is applied throughout the image, with small blur radius
near the foveation origin, which increases as we go further into the periphery.
This causes the image to appear sharp near the origin while blurry toward the
outer edges.

Testing has revealed that BlurFoveator is not as efficient as PyrFoveator or
LPTFoveator but can still achieve the desired result.
*/

class BlurFoveator : public Foveator
{
private:
  //! An inner class that defines a gaussian blur matrix
  /*! This class is used for obtaining blurred RGB pixels at different blur
    radii. */
  class GaussianBlurMatrix
  {
  private:
    float *weights;
    int halfFilter;
    int size;

  public:
    //! Construct a blur matrix with specified filter size
    GaussianBlurMatrix( int filterSize );
    //! Construct a GaussianBlurMatrix from another
    GaussianBlurMatrix( const GaussianBlurMatrix& gbm );
    //! Construct a GaussianBlurMatrix equal to another
    GaussianBlurMatrix& operator=( const GaussianBlurMatrix& gbm );
    //! Destructor
    ~GaussianBlurMatrix();

    //! Calculate a blurred pixel
    /* Returns a blurred RGB pixel after Gaussian filter is applied.
       @param x horizontal coordinate of pixel to be blurred
       @param y vertical coordinate of pixel to be blurred
       @param img image from which pixel is taken
    */
    PixRGB<byte> blurPixel( int x, int y,
                            const Image< PixRGB<byte> >& img ) const;
    //! Set the Gaussian blur radius
    void setBlurRadius( const float& r );
  };

  GaussianBlurMatrix **gbms;
  int matrices;
  static float radius( int x, int y, int area );

public:
  //! Construct a BlurFoveator for an image
  /*! This constructor will create a BlurFoveator that handles img and all
    other images of the same dimensions.
    @param img image to be foveated
    @param filterSize size of blur filter to be applied
  */
  BlurFoveator( const Image< PixRGB<byte> >& img, int filterSize );
  //! Construct a BlurFoveator from another
  BlurFoveator( const BlurFoveator& bf );
  //! Construct a BlurFoveator equal to another
  BlurFoveator& operator=( const BlurFoveator& bf );
  //! Destructor
  ~BlurFoveator();

  //! Perform blur foveation and return resulting image
  Image< PixRGB<byte> > foveate( void );

  //! Perform one blur foveation without a BlurFoveator
  /*! This function is for foveating ONE image with a call such as
    BlurFoveator::foveate().  It is not optimized for foveating multiple
    images.
    @param img image to be foveated
    @param filterSize size of blur filter to be applied
    @param x horizontal coordinate of origin
    @param y vertical coordinate of origin
  */
  static Image< PixRGB<byte> > foveate( const Image< PixRGB<byte> >& img,
                                        int filterSize, int x, int y );
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
