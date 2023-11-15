/*!@file Foveator/LPTFoveator.H Foveator class that performs log polar transform */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/LPTFoveator.H $
// $Id: LPTFoveator.H 9412 2008-03-10 23:10:15Z farhan $

#ifndef LPTFOVEATOR_H_DEFINED
#define LPTFOVEATOR_H_DEFINED

#include "Foveator/Foveator.H"
#include <cmath>
#include <map>
#include <utility>

//! A class that does space-variant processing with log polar transform
/*! A log-polar transform (LPT) maps pixels of the original image to a smaller
"map" image that holds pixel data.  The LPT transforms rectangular \f$(x,y)\f$
coordinates into \f$(r,\theta)\f$, where r is a function of the log of the
radius from the point to the origin, and \f$\theta\f$ is the angle from the
horizontal axis.

For points close to the origin, the result is that the LPT mapping is almost
one-to-one, while in the periphery many pixels may correspond to a single
ordered pair of \f$(r,\theta)\f$, resulting in some degree of data loss.  To
display the foveated image an inverse LPT must be performed with the invLPT()
set of methods.

LPTFoveator is very fast compared with BlurFoveator and PyrFoveator, and
is useful for compression.  However, there is a significant amount of
blockiness in the periphery in the foveated image, and it seems that noise can
have a serious effect on image quality.
*/

class LPTFoveator : public Foveator
{
private:
  int lptWidth;
  int lptHeight;
  double logMultiplier;

  //! a function object for the key compare in multimap
  struct ltPoint2D
  {
    bool operator()( const Point2D<int> pt1, const Point2D<int> pt2 ) const
    {
      if( pt1.j < pt2.j )
        return true;
      else if( pt1.j > pt2.j )
        return false;
      else
        {
          return( pt1.i < pt2.i );
        }
    }
  };

  // to make life simpler for the coder:
  typedef std::multimap< Point2D<int>, Point2D<int>, ltPoint2D > Point2DMap;
  typedef Point2DMap::const_iterator Point2DMapPtr;

  Point2DMap lptMap; //!< our log-polar mapping

  //! Radius calculation function based on distance between point and origin
  inline double radius( int x, int y );
  /*! Angle calculation function based on angle from origin (positive x-axis)
    to point */
  inline double angle( int x, int y );

public:
  //! Construct a LPTFoveator for img
  /*! This constructor will create a LPTFoveator that handles img and all
    other images of the same dimensions.
    @param img image to be foveated
    @param lptW width of log-polar map image, controls radial detail
    @param lptH height of log-polar map image, controls angular detail
  */
  LPTFoveator( const Image< PixRGB<byte> >& img, int lptW, int lptH );

  //! Destructor
  ~LPTFoveator();

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

  //! Perform log-polar foveation and return resulting image
  Image< PixRGB<byte> > foveate( void );
  //! Perform log-polar transform and return map image
  Image< PixRGB<byte> > getLPT( void );
  //! Perform inverse log-polar transform on a map image to get foveated image
  /*! This function can do an inverse log-polar transform on an image returned
    by LPTFoveator::getLPT().  It will be important that this function must be
    called from the same LPTFoveator object that created the map image, or
    from one with the same configuration parameters. Also, the origin must not
    have been changed since the map image was generated.
    @param img map image (source from which foveated image will be created)
   */
  Image< PixRGB<byte> > invLPT( const Image< PixRGB<byte> >& img );

  //! Perform one log-polar foveation without a LPTFoveator
  /*! This function is for foveating ONE image with a call such as
    LPTFoveator::foveate().  It is not optimized for foveating multiple
    images.
    @param img image to be foveated
    @param lptW width of log-polar map image, controls radial detail
    @param lptH height of log-polar map image, controls angular detail
    @param x horizontal coordinate of origin
    @param y vertical coordinate of origin
    @param getMap set this to true if you want to have this function return
    map image rather than foveated image
  */
  static Image< PixRGB<byte> > foveate( const Image< PixRGB<byte> >& img,
                                        int lptW, int lptH, int x, int y,
                                        bool getMap = false );
  //! Get foveated image from log-polar map image
  /*! This function can be used to perform an inverse log-polar transform
    without a LPTFoveator.  It is not optimized for performing this operation
    on multiple images, but it would be faster to use this function on an map
    image than to call LPTFoveator::foveate() again with getMap = true.
    @param img image to be foveated
    @param w width of original image (before LPT)
    @param h height of original image (before LPT)
    @param x horizontal coordinate of origin
    @param y vertical coordinate of origin
  */
  static Image< PixRGB<byte> > invLPT( const Image< PixRGB<byte> >& img,
                                       int w, int h, int x, int y );
};

// ######################################################################
// ########## Inlined methods:
// ######################################################################

inline double LPTFoveator::radius( int x, int y )
{
  return( logMultiplier * log( x*x + y*y + 1.0 ) + 0.5 );
}

inline double LPTFoveator::angle( int x, int y )
{
  if( y == 0 )
    {
      return( ( x >= 0 ) ?
              0.0 : 0.5 * ( lptHeight - 1 ) );
    }
  if( x == 0 )
    {
      return( ( y > 0 ) ?
              0.25 * ( lptHeight - 1 ) : 0.75 * ( lptHeight - 1 ) );
    }
  return( ( M_PI + atan2( -1.0 * y, -1.0 * x ) ) /
          ( 2 * M_PI ) * ( lptHeight - 1 ) + 0.5 );
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
