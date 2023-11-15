/*!@file Foveator/BlurFoveator.C Foveator class that performs progressive blurring */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/BlurFoveator.C $
// $Id: BlurFoveator.C 4663 2005-06-23 17:47:28Z rjpeters $

#include "Foveator/BlurFoveator.H"

#include "Image/Pixels.H"

#include <algorithm> // for std::max()
#include <cmath>

// ######################################################################

// constructor with initialization of original image
BlurFoveator::BlurFoveator( const Image< PixRGB< byte > >& img,
                            int filterSize ) : Foveator( img )
{
  // find the greater of two dimensions
  int dMax = std::max( width, height );

  // generate blur filters
  matrices = (dMax+1) * dMax / 2 - 1;
  typedef GaussianBlurMatrix* GaussianBlurMatrixPtr;
  gbms = new GaussianBlurMatrixPtr[matrices];
  int k = -1;
  for( int i = 0; i < dMax; i++ )
    {
      for( int j = 0; j <= i; j++ )
        {
          if( k > -1 )
            {
              gbms[k] = new GaussianBlurMatrix( filterSize );
              gbms[k]->setBlurRadius( radius( i, j, width * height ) );
            }
          k++;
        }
    }
}

// copy constructor
BlurFoveator::BlurFoveator( const BlurFoveator& bf ) :
  Foveator( bf.original )
{
  matrices = bf.matrices;
  typedef GaussianBlurMatrix* GaussianBlurMatrixPtr;
  gbms = new GaussianBlurMatrixPtr[matrices];
  for( int i = 0; i < matrices; i++ )
    {
      gbms[i] = new GaussianBlurMatrix( *(bf.gbms[i]) );
    }
}

// assignment operator
BlurFoveator& BlurFoveator::operator=( const BlurFoveator& bf )
{
  BlurFoveator *newbf = new BlurFoveator( bf );
  return *newbf;
}

// destructor
BlurFoveator::~BlurFoveator()
{
  delete [] gbms;
}

// ######################################################################

// constructor for GaussianBlurMatrix
BlurFoveator::GaussianBlurMatrix::GaussianBlurMatrix( int filterSize )
{
  if( filterSize % 2 == 0)
    {
      LERROR( "Blur filter size should be positive odd integer!" );
      LINFO( "Blur foveation will be unstable/unpredictable!" );
    }
  halfFilter = filterSize/2 + 1;
  size = (halfFilter+1) * halfFilter / 2;
  weights = new float[size];
}

// copy constructor for GaussianBlurMatrix
BlurFoveator::GaussianBlurMatrix::
GaussianBlurMatrix( const GaussianBlurMatrix& gbm )
{
  halfFilter = gbm.halfFilter;
  size = gbm.size;
  weights = new float[size];
  for( int i = 0; i < size; i++ )
    {
      weights[i] = gbm.weights[i];
    }
}

// assignment operator
BlurFoveator::GaussianBlurMatrix& BlurFoveator::GaussianBlurMatrix::
operator=( const GaussianBlurMatrix& gbm )
{
  GaussianBlurMatrix *newgbm = new GaussianBlurMatrix( gbm );
  return *newgbm;
}

// destructor for GaussianBlurMatrix
BlurFoveator::GaussianBlurMatrix::~GaussianBlurMatrix()
{
  delete [] weights;
}

// calculate a blurred pixel
PixRGB<byte> BlurFoveator::GaussianBlurMatrix::
blurPixel( int x, int y, const Image< PixRGB<byte> >& img ) const
{
  float newR = 0.0f;
  float newG = 0.0f;
  float newB = 0.0f;
  float denom = 0.0f;

  int k = 0;
  for( int i = 0; i < halfFilter; i++ )
    {
      for( int j = 0; j <= i; j++ )
        {
          if( !k )
            {
              PixRGB<byte> pix = img.getVal( x, y );
              newR += pix.red();
              newG += pix.green();
              newB += pix.blue();
              denom++;
            }
          else
            {
              if( img.coordsOk( x + i, y + j ) )
                {
                  PixRGB<byte> pix = img.getVal( x + i, y + j );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x + j, y + i ) )
                {
                  PixRGB<byte> pix = img.getVal( x + j, y + i );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x - i, y + j ) )
                {
                  PixRGB<byte> pix = img.getVal( x - i, y + j );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x - j, y + i ) )
                {
                  PixRGB<byte> pix = img.getVal( x - j, y + i );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x - i, y - j ) )
                {
                  PixRGB<byte> pix = img.getVal( x - i, y - j );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x - j, y - i ) )
                {
                  PixRGB<byte> pix = img.getVal( x - j, y - i );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x + i, y - j ) )
                {
                  PixRGB<byte> pix = img.getVal( x + i, y - j );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
              if( img.coordsOk( x + j, y - i ) )
                {
                  PixRGB<byte> pix = img.getVal( x + j, y - i );
                  newR += weights[k] * pix.red();
                  newG += weights[k] * pix.green();
                  newB += weights[k] * pix.blue();
                  denom += weights[k];
                }
            }
          k++;
        }
    }

  newR /= denom;
  newG /= denom;
  newB /= denom;
  PixRGB<byte> newPixel( newR, newG, newB );
  return newPixel;
}

// set Gaussian blur radius
void BlurFoveator::GaussianBlurMatrix::setBlurRadius( const float& r )
{
  int k = 0;
  for( int i = 0; i < halfFilter; i++ )
    {
      for( int j = 0; j <= i; j++ )
        {
          weights[k] = (float)( exp( -( (i*i+j*j)/(2*r*r) ) ) );
          k++;
        }
    }
}

// ######################################################################

// Radius calculation function based on distance between point and origin
// Should be monotonic increasing
float BlurFoveator::radius( int x, int y, int area )
{
  return( 10.0f * ( x*x + y*y ) / area + 0.5f );
}

// Blur-foveation method, returns foveated image
Image< PixRGB<byte> > BlurFoveator::foveate( void )
{
  Image< PixRGB<byte> > fovImg( width, height, NO_INIT );

  // determine maximum distance to an edge
  int dMax = std::max( std::max( origin.i, width - origin.i ),
                       std::max( origin.j, height - origin.j ) );

  int k = -1;
  for( int i = 0; i < dMax; i++ )
    {
      for( int j = 0; j <= i; j++ )
        {
          // check if we are operating on origin pixel
          if( k == -1 )
            {
              fovImg.setVal( origin.i, origin.j,
                             original.getVal( origin.i, origin.j ) );
            }
          else
            {
              if( fovImg.coordsOk( origin.i + i, origin.j + j ) )
                {
                  fovImg.setVal( origin.i + i, origin.j + j,
                                 gbms[k]->blurPixel( origin.i + i,
                                                     origin.j + j,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i + j, origin.j + i ) )
                {
                  fovImg.setVal( origin.i + j, origin.j + i,
                                 gbms[k]->blurPixel( origin.i + j,
                                                     origin.j + i,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i - i, origin.j + j ) )
                {
                  fovImg.setVal( origin.i - i, origin.j + j,
                                 gbms[k]->blurPixel( origin.i - i,
                                                     origin.j + j,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i - j, origin.j + i ) )
                {
                  fovImg.setVal( origin.i - j, origin.j + i,
                                 gbms[k]->blurPixel( origin.i - j,
                                                     origin.j + i,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i - i, origin.j - j ) )
                {
                  fovImg.setVal( origin.i - i, origin.j - j,
                                 gbms[k]->blurPixel( origin.i - i,
                                                     origin.j - j,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i - j, origin.j - i ) )
                {
                  fovImg.setVal( origin.i - j, origin.j - i,
                                 gbms[k]->blurPixel( origin.i - j,
                                                     origin.j - i,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i + i, origin.j - j ) )
                {
                  fovImg.setVal( origin.i + i, origin.j - j,
                                 gbms[k]->blurPixel( origin.i + i,
                                                     origin.j - j,
                                                     original ) );
                }
              if( fovImg.coordsOk( origin.i + j, origin.j - i ) )
                {
                  fovImg.setVal( origin.i + j, origin.j - i,
                                 gbms[k]->blurPixel( origin.i + j,
                                                     origin.j - i,
                                                     original ) );
                }
            }
          k++;
        }
    }

  return fovImg;
}

// ######################################################################

Image< PixRGB<byte> > BlurFoveator::
foveate( const Image< PixRGB<byte> >& img, int filterSize, int x, int y )
{
  // generate an image of same dimensions
  int width = img.getWidth();
  int height = img.getHeight();
  Image< PixRGB< byte > > fovImg( width, height, NO_INIT );

  // generate a Gaussian blur filter of the appropriate size
  GaussianBlurMatrix gbm( filterSize );

  // determine maximum distance to an edge
  int dMax = std::max( std::max( x, width - x ),
                       std::max( y, height - y ) );

  for( int i = 0; i < dMax; i++ )
    {
      for( int j = 0; j <= i; j++ )
        {
          // check if we are operating on the origin pixel
          if( !i && !j )
            {
              fovImg.setVal( x, y, img.getVal( x, y ) );
            }
          else
            {
              gbm.setBlurRadius( radius( i, j, width * height ) );
              if( fovImg.coordsOk( x + i, y + j ) )
                {
                  fovImg.setVal( x + i, y + j,
                                 gbm.blurPixel( x + i, y + j, img ) );
                }
              if( fovImg.coordsOk( x + j, y + i ) )
                {
                  fovImg.setVal( x + j, y + i,
                                 gbm.blurPixel( x + j, y + i, img ) );
                }
              if( fovImg.coordsOk( x - i, y + j ) )
                {
                  fovImg.setVal( x - i, y + j,
                                 gbm.blurPixel( x - i, y + j, img ) );
                }
              if( fovImg.coordsOk( x - j, y + i ) )
                {
                  fovImg.setVal( x - j, y + i,
                                 gbm.blurPixel( x - j, y + i, img ) );
                }
              if( fovImg.coordsOk( x - i, y - j ) )
                {
                  fovImg.setVal( x - i, y - j,
                                 gbm.blurPixel( x - i, y - j, img ) );
                }
              if( fovImg.coordsOk( x - j, y - i ) )
                {
                  fovImg.setVal( x - j, y - i,
                                 gbm.blurPixel( x - j, y - i, img ) );
                }
              if( fovImg.coordsOk( x + i, y - j ) )
                {
                  fovImg.setVal( x + i, y - j,
                                 gbm.blurPixel( x + i, y - j, img ) );
                }
              if( fovImg.coordsOk( x + j, y - i ) )
                {
                  fovImg.setVal( x + j, y - i,
                                 gbm.blurPixel( x + j, y - i, img ) );
                }
            }
        }
    }

  return fovImg;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
