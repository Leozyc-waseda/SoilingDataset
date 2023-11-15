/*!@file Foveator/PyrFoveator.C Foveator class that builds dyadic pyramid */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/PyrFoveator.C $
// $Id: PyrFoveator.C 7293 2006-10-20 18:49:55Z rjpeters $

#include "Foveator/PyrFoveator.H"

#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"

// ######################################################################

// constructor with initialization of original image
PyrFoveator::PyrFoveator( const Image< PixRGB<byte> >& img,
                          int filterSize ) : Foveator( img ),
                                             builder( filterSize ),
                                             baseRect( 1, 1 )
{
}

// copy constructor
PyrFoveator::PyrFoveator( const PyrFoveator& pf ) : Foveator( pf.original ),
                                                    builder( pf.builder )
{
  baseRect = pf.baseRect;
}

// assignment operator
PyrFoveator& PyrFoveator::operator=( const PyrFoveator& pf )
{
  PyrFoveator *newpf = new PyrFoveator( pf );
  return *newpf;
}

// destructor
PyrFoveator::~PyrFoveator()
{
}

// ######################################################################

// set base rectangle size
// at center of foveation, all rects will be multiples of this one
void PyrFoveator::setBaseRect( Dims& d )
{
  baseRect = d;
}

void PyrFoveator::setBaseRect( int x, int y )
{
  Dims rect(x,y);
  baseRect = rect;
}

// ######################################################################

Image< PixRGB<byte> > PyrFoveator::foveate( void )
{
  // determine necessary pyramid depth
  int expW = baseRect.w();
  int expH = baseRect.h();
  int depth = 0;
  while( ( origin.i + expW < width ) || ( origin.i - expW > 0 ) ||
         ( origin.j + expW < height ) || ( origin.j - expW > 0 ) )
    {
      depth++;
      expW *= 2;
      expH *= 2;
    }

  // build pyramid
  ImageSet< PixRGB<byte> > pyramid( depth );
  pyramid = builder.build( original, 0, depth );
  for( int d = 0; d < depth; d++ )
    pyramid[d] = rescale( pyramid[d], width, height );

  // pile the images on one another like a pyramid
  Image< PixRGB<byte> > fovImg( pyramid[depth-1] );
  for( int d = depth - 2; d >= 0; d-- )
    {
      int xOffset = baseRect.w();
      int yOffset = baseRect.h();
      for( int p = 0; p < d; p++ )
        {
          xOffset *= 2;
          yOffset *= 2;
        }
      for( int x = origin.i - xOffset; x < origin.i + xOffset; x++ )
        {
          for( int y = origin.j - yOffset; y < origin.j + yOffset; y++ )
            {
              if( fovImg.coordsOk( x, y ) )
                {
                  fovImg.setVal( x, y, pyramid[d].getVal( x, y ) );
                }
            }
        }
    }

  // return the image
  return fovImg;
}

// ######################################################################

Image< PixRGB<byte> > PyrFoveator::foveate( const Image< PixRGB<byte> >& img,
                                            int filterSize, int baseRectWidth,
                                            int baseRectHeight, int x, int y )
{
  // determine necessary pyramid depth
  int expW = baseRectWidth;
  int expH = baseRectHeight;
  int depth = 0;
  while( ( x + expW < img.getWidth() ) || ( x - expW > 0 ) ||
         ( y + expW < img.getHeight() ) || ( y - expW > 0 ) )
    {
      depth++;
      expW *= 2;
      expH *= 2;
    }

  // build pyramid
  GaussianPyrBuilder< PixRGB<byte> > builder( filterSize );
  ImageSet< PixRGB<byte> > pyramid( depth );
  pyramid = builder.build( img, 0, depth );
  for( int d = 0; d < depth; d++ )
    pyramid[d] = rescale( pyramid[d], img.getWidth(), img.getHeight() );

  // pile the images on one another like a pyramid
  Image< PixRGB<byte> > fovImg( pyramid[depth-1] );
  for( int d = depth - 2; d >= 0; d-- )
    {
      int xOffset = baseRectWidth;
      int yOffset = baseRectHeight;
      for( int p = 0; p < d; p++ )
        {
          xOffset *= 2;
          yOffset *= 2;
        }
      for( int xx = x - xOffset; xx < x + xOffset; xx++ )
        {
          for( int yy = y - yOffset; yy < y + yOffset; yy++ )
            {
              if( fovImg.coordsOk( xx, yy ) )
                {
                  fovImg.setVal( xx, yy, pyramid[d].getVal( xx, yy ) );
                }
            }
        }
    }

  // return the image
  return fovImg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
