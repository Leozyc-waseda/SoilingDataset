/*!@file Foveator/Foveator.H An abstract class for space-variant image processing */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/Foveator.H $
// $Id: Foveator.H 9412 2008-03-10 23:10:15Z farhan $

#ifndef FOVEATOR_H_DEFINED
#define FOVEATOR_H_DEFINED

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Point2D.H"

//! An abstract class for space-variant image processing
/*! Space-variant image processing concentrates image quality and resolution
in areas of interest.  Human vision has high resolution at the center, but it
decreases as we move further out into the periphery.  Through the process of
foveation, we attempt to simulate this type of vision.

Classes used for foveation should inherit this as a base.  The foveation
operation is performed through pure virtual function foveate().  Different
foveation techniques and algorithms are implemented in BlurFoveator,
PyrFoveator, and LPTFoveator.
*/

class Foveator
{
protected:
  Image< PixRGB< byte > > original; //!< image to be foveated
  int width;                        //!< width of original image
  int height;                       //!< height of original image
  Point2D<int> origin;                   //!< center of foveation

public:
  //! Constructor
  /*! Create a Foveator object to operate on image argument
    @param img image to be foveated
  */
  Foveator( const Image< PixRGB< byte > >& img );
  //! Destructor
  virtual ~Foveator();

  //! Returns the current center of foveation
  Point2D<int> getOrigin( void ) const;
  //! Returns width of image being foveated
  int getWidth( void ) const;
  //! Returns height of image being foveated
  int getHeight( void ) const;

  //! Change the image to be foveated
  /*! Change the image to be foveated to another one of the same size
    @param img new image to be used, must have same dimensions as original
  */
  void changeImage( const Image< PixRGB< byte > >& img );

  //! Set origin point (center of foveation)
  /*! Set origin point to Point2D<int> argument, returns true if successful
    @param pt center of foveation
  */
  virtual bool setOrigin( Point2D<int> pt );
  //! Set origin point (center of foveation)
  /*! Set origin point to two int arguments, returns true if successful
    @param x horizontal coordinate of origin
    @param y vertical coordinate of origin
  */
  virtual bool setOrigin( int x, int y );

  //! Foveation function, returns foveated image
  virtual Image< PixRGB< byte> > foveate( void ) = 0;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
