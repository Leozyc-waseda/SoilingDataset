/*!@file Foveator/Foveator.C An abstract class for space-variant image processing */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/Foveator.C $
// $Id: Foveator.C 9412 2008-03-10 23:10:15Z farhan $

#include "Foveator/Foveator.H"

#include "Image/Pixels.H"
#include "Util/log.H"

// ######################################################################

// constructor
Foveator::Foveator( const Image< PixRGB< byte > >& img ) :
  original( img ),
  width( img.getWidth() ), height( img.getHeight() ),
  origin( width/2, height/2 )
{
}

// destructor
Foveator::~Foveator()
{
}

// ######################################################################

// get coordinates of origin
Point2D<int> Foveator::getOrigin( void ) const
{
  return origin;
}

// get width
int Foveator::getWidth( void ) const
{
  return width;
}

// get height
int Foveator::getHeight( void ) const
{
  return height;
}

// ######################################################################

// change the image to be foveated
void Foveator::changeImage( const Image< PixRGB< byte > >& img )
{
  if( width == img.getWidth() && height == img.getHeight() )
    {
      original = img;
    }
  else
    {
      LERROR( "New image must have same dimensions as original." );
      LERROR( "Image has not been changed." );
    }
}

// ######################################################################

// set origin to Point2D<int>
bool Foveator::setOrigin( Point2D<int> pt )
{
  if( original.coordsOk( pt ) )
    {
      origin = pt;
      return true;
    }
  else
    {
      LERROR( "Center of foveation needs to be inside image bounds." );
      return false;
    }
}

// set origin to integer coordinates
bool Foveator::setOrigin( int x, int y )
{
  if( original.coordsOk( x, y ) )
    {
      origin.i = x;
      origin.j = y;
      return true;
    }
  else
    {
      LERROR( "Center of foveation needs to be inside image bounds." );
      return false;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
