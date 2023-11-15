/*!@file Foveator/LPTFoveator.C Foveator class that performs log polar transform */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Foveator/LPTFoveator.C $
// $Id: LPTFoveator.C 9412 2008-03-10 23:10:15Z farhan $

#include "Foveator/LPTFoveator.H"

#include "Image/Pixels.H"

#include <algorithm> // for std::max()

// ######################################################################

// constructor with initialization of original image

LPTFoveator::LPTFoveator( const Image< PixRGB<byte> >& img,
                          int lptW, int lptH) : Foveator( img ),
                                                lptWidth( lptW ),
                                                lptHeight( lptH )
{
  if( lptW <= 0 || lptH <= 0 )
    LERROR( "LPTFoveator map image must have positive dimensions." );
  setOrigin( width / 2, height / 2 );
}

// destructor

LPTFoveator::~LPTFoveator()
{
  lptMap.clear();
}

// ######################################################################

// set origin and recalculate the LPT mapping

bool LPTFoveator::setOrigin( Point2D<int> pt )
{
  if( Foveator::setOrigin( pt ) )
    {
      // find log multiplier
      int maxX = std::max( origin.i, width - origin.i );
      int maxY = std::max( origin.j, height - origin.j );
      logMultiplier = ( lptWidth - 1 ) / log( maxX*maxX + maxY*maxY + 1.0 );

      // calculate LPT pixel mapping
      lptMap.clear();
      Point2D<int> mapPt;
      Point2D<int> originalPt;
      for( int x = 0; x < width; x++ )
        {
          for( int y = 0; y < height; y++ )
            {
              originalPt.i = x;
              originalPt.j = y;
              mapPt.i = (int)radius( x - origin.i, y - origin.j );
              mapPt.j = (int)angle( x - origin.i, y - origin.j );
              lptMap.insert( std::pair< Point2D<int>, Point2D<int> >
                             ( mapPt, originalPt ) );
            }
        }

      // send positive feedback
      return true;
    }
  else
    return false;
}

bool LPTFoveator::setOrigin( int x, int y )
{
  Point2D<int> pt( x, y );
  return setOrigin( pt );
}

// ######################################################################

Image< PixRGB<byte> > LPTFoveator::foveate( void )
{
  Image< PixRGB<byte> > fovImg = invLPT( getLPT() );
  return fovImg;
}

// ######################################################################

Image< PixRGB<byte> > LPTFoveator::getLPT( void )
{
  Image< PixRGB<byte> > lptImg( lptWidth, lptHeight, NO_INIT );
  for( int x = 0; x < lptWidth; x++ )
    {
      for( int y = 0; y < lptHeight; y++ )
        {
          Point2D<int> pt( x, y );
          Point2DMapPtr pmp = lptMap.lower_bound( pt );
          int count = 0;
          int rAcc = 0;
          int gAcc = 0;
          int bAcc = 0;
          while( pmp != lptMap.upper_bound( pt ) )
            {
              // accumulation of RGB values
              rAcc += original.getVal( (*pmp).second ).red();
              gAcc += original.getVal( (*pmp).second ).green();
              bAcc += original.getVal( (*pmp).second ).blue();
              count++;
              pmp++;
            }
          if( count > 1 )
            {
              rAcc /= count;
              gAcc /= count;
              bAcc /= count;
            }
          PixRGB<byte> pix( rAcc, gAcc, bAcc );
          lptImg.setVal( pt, pix );
        }
    }
  return lptImg;
}

// ######################################################################

Image< PixRGB<byte> > LPTFoveator::invLPT( const Image< PixRGB<byte> >& img )
{
  Image< PixRGB<byte> > invImg( width, height, NO_INIT );
  for( Point2DMapPtr pmp = lptMap.begin(); pmp != lptMap.end(); pmp++ )
    {
      invImg.setVal( (*pmp).second, img.getVal( (*pmp).first ) );
    }
  return invImg;
}

// ######################################################################

Image< PixRGB<byte> > LPTFoveator::foveate( const Image< PixRGB<byte> >& img,
                                            int lptW, int lptH, int x, int y,
                                            bool getMap )
{
  // find log multiplier
  int maxX = std::max( x, img.getWidth() - x );
  int maxY = std::max( y, img.getHeight() - y );
  double logMult = ( lptW - 1 ) / log( maxX * maxX + maxY * maxY + 1.0 );

  // declare and allocate accumulators and counter
  int **R_acc;
  int **G_acc;
  int **B_acc;
  int **counter;

  typedef int* intptr;
  R_acc = new intptr[lptW];
  G_acc = new intptr[lptW];
  B_acc = new intptr[lptW];
  counter = new intptr[lptW];
  for( int i = 0; i < lptW; i++ )
    {
      R_acc[i] = new int[lptH];
      G_acc[i] = new int[lptH];
      B_acc[i] = new int[lptH];
      counter[i] = new int[lptH];
    }

  // initialize accumulators and counter to all zeros
  for( int i = 0; i < lptW; i++ )
    {
      for( int j = 0; j < lptH; j++ )
        {
          R_acc[i][j] = 0;
          G_acc[i][j] = 0;
          B_acc[i][j] = 0;
          counter[i][j] = 0;
        }
    }

  // accumulate and count LPT pixels
  for( int i = 0; i < img.getWidth(); i++ )
    {
      for( int j = 0; j < img.getHeight(); j++ )
        {
          // make coordinates relative to origin
          int xx = i - x;
          int yy = j - y;

          // determine r
          int r = int( logMult * log( xx * xx + yy * yy + 1.0 ) + 0.5 );
          // determine theta
          int theta;
          if( yy == 0 )
            {
              theta = (int)( ( xx >= 0 ) ?
                             0.0 : 0.5 * ( lptH - 1 ) );
            }
          else if( xx == 0 )
            {
              theta = (int)( ( yy > 0 ) ?
                             0.25 * ( lptH - 1 ) : 0.75 * ( lptH - 1 ) );
            }
          else
            {
              theta = (int)( ( M_PI + atan2( -1.0 * yy, -1.0 * xx ) ) /
                ( 2 * M_PI ) * ( lptH - 1 ) + 0.5 );
            }

          // add pixel values
          R_acc[r][theta] += ( img.getVal( i, j ) ).red();
          G_acc[r][theta] += ( img.getVal( i, j ) ).green();
          B_acc[r][theta] += ( img.getVal( i, j ) ).blue();
          counter[r][theta]++;
        }
    }

  // write pixels to image
  Image< PixRGB<byte> > mapImage( lptW, lptH, NO_INIT );
  for( int r = 0; r < lptW; r++ )
  {
    for( int theta = 0; theta < lptH; theta++ )
    {
      if( counter[r][theta] > 1 )
        {
          // divide accumulators by counter matrix
          R_acc[r][theta] /= counter[r][theta];
          G_acc[r][theta] /= counter[r][theta];
          B_acc[r][theta] /= counter[r][theta];
        }
      PixRGB<byte> pix( R_acc[r][theta], G_acc[r][theta], B_acc[r][theta] );
      mapImage.setVal( r, theta, pix );
    }
  }

  // free memory
  delete [] R_acc;
  delete [] G_acc;
  delete [] B_acc;
  delete [] counter;

  if( getMap )
    return mapImage;
  else
    return ( invLPT( mapImage, img.getWidth(), img.getHeight(), x, y ) );
}

// ######################################################################

Image< PixRGB<byte> > LPTFoveator::invLPT( const Image< PixRGB<byte> >& img,
                                           int w, int h, int x, int y )
{
  // find log multiplier
  int maxX = std::max( x, w - x );
  int maxY = std::max( y, h - y );
  double logMult = ( img.getWidth() - 1 ) /
    log( maxX * maxX + maxY * maxY + 1.0 );

  Image< PixRGB<byte> > invImage( w, h, NO_INIT );

  // write to image
  for( int i = 0; i < w; i++ )
  {
    for( int j = 0; j < h; j++ )
    {
      // make coordinates relative to origin
      int xx = i - x;
      int yy = j - y;

      // determine r
      int r = int( logMult * log( xx * xx + yy * yy + 1.0 ) + 0.5 );
      // determine theta
      int theta;
      if( yy == 0 )
        {
          theta = (int)( ( xx >= 0 ) ?
                         0.0 : 0.5 * ( img.getHeight() - 1 ) );
        }
      else if( xx == 0 )
        {
          theta = (int)( ( yy > 0 ) ?
                         0.25 * ( img.getHeight() - 1 ) :
                         0.75 * ( img.getHeight() - 1 ) );
        }
      else
        {
          theta = (int)( ( M_PI + atan2( -1.0 * yy, -1.0 * xx ) ) /
                         ( 2 * M_PI ) * ( img.getHeight() - 1 ) + 0.5 );
        }

      invImage.setVal( i, j, img.getVal( r, theta ) );
    }
  }
  return invImage;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
