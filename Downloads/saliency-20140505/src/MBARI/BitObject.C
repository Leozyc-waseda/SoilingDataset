/*!@file MBARI/BitObject.C class that holds an object defined by a shape in a
bit image */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/BitObject.C $
// $Id: BitObject.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "MBARI/BitObject.H"

#include "Image/CutPaste.H"    // for crop()
#include "Image/IO.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "Raster/GenericFrame.H"
#include "Raster/PnmParser.H"
#include "Raster/PnmWriter.H"
#include "Util/Assert.H"
#include "Util/MathFunctions.H"

#include <cmath>
#include <istream>
#include <ostream>

// ######################################################################
BitObject::BitObject()
{
  freeMem();
}

// ######################################################################
BitObject::BitObject(const Image<byte>& img, const Point2D<int> location,
                     const byte threshold)
{
  reset(img, location, threshold);
}

// ######################################################################
BitObject::BitObject(const Image<byte>& img)
{
  reset(img);
}

// ######################################################################
BitObject::BitObject(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
Image<byte> BitObject::reset(const Image<byte>& img, const Point2D<int> location,
                             const byte threshold)
{
  ASSERT(img.initialized());

  // first, reset everything to defaults
  freeMem();

  // now, flood img to get the object
  Image<byte> dest;
  int area = floodCleanBB(img, dest, location, threshold,
                          byte(1), itsBoundingBox);

  // no object found? return -1
  if (area == -1)
    {
      itsCentroidXY.reset(location);
      return Image<byte>();
    }

  // set the dimensions of the original image
  itsImageDims = img.getDims();

  // crop the object mask from the flooding destination
  itsObjectMask = crop(dest, itsBoundingBox);

  // get the area, the centroid, and the bounding box
  std::vector<float> sumx, sumy;
  itsArea = (int)sumXY(itsObjectMask, sumx, sumy);
  if (area != itsArea)
    LFATAL("area %i doesn't match the one from flooding %i", itsArea, area);

  int firstX, lastX, firstY, lastY;
  float cX, cY;
  bool success = (getCentroidFirstLast(sumx, cX, firstX, lastX) |
                  getCentroidFirstLast(sumy, cY, firstY, lastY));
  itsCentroidXY.reset(cX,cY);

  if (!success) LFATAL("determining the centroid failed");

  if ((firstX != 0) || (lastX != itsObjectMask.getWidth()-1) ||
      (firstY != 0) || (lastY != itsObjectMask.getHeight()-1))
    LFATAL("boundary box doesn't match the one from flooding");

  itsCentroidXY += Vector2D(itsBoundingBox.left(),itsBoundingBox.top());

  return dest;
}

// ######################################################################
int BitObject::reset(const Image<byte>& img)
{
  ASSERT(img.initialized());

  // first, reset everything to defaults
  freeMem();

  // set the dimensions of the original image
  itsImageDims = img.getDims();

  // get the area, the centroid, and the bounding box
  std::vector<float> sumx, sumy;
  itsArea = (int)sumXY(img, sumx, sumy);

  if (itsArea == 0)
    {
      freeMem();
      return -1;
    }

  int firstX, lastX, firstY, lastY;
  float cX, cY;
  bool success = (getCentroidFirstLast(sumx, cX, firstX, lastX) |
                  getCentroidFirstLast(sumy, cY, firstY, lastY));
  itsCentroidXY.reset(cX,cY);

  if (!success)
    {
      freeMem();
      return -1;
    }

  itsBoundingBox = Rectangle::tlbrI(firstY, firstX, lastY, lastX);

  // cut out the object mask
  itsObjectMask = crop(img, itsBoundingBox);

  //LINFO("BB: size: %i; %s; dims: %s",itsBoundingBox.width()*itsBoundingBox.height(),
  //    toStr(itsBoundingBox).c_str(),toStr(itsObjectMask.getDims()).c_str());

  return itsArea;
}

// ######################################################################
void BitObject::computeSecondMoments()
{
  ASSERT(isValid());

  int w = itsObjectMask.getWidth();
  int h = itsObjectMask.getHeight();

  // The bounding box is stored in image coordinates, and so is the centroid. For
  // computing the second moments, however we need the centroid in object coords.
  float cenX = itsCentroidXY.x() - itsBoundingBox.left();
  float cenY = itsCentroidXY.y() - itsBoundingBox.top();

  // compute the second moments
  std::vector<float> diffX(w), diffX2(w), diffY(h), diffY2(h);
  for (int y = 0; y < h; ++y)
    {
      diffY[y] = y - cenY;
      diffY2[y] = diffY[y] * diffY[y];
    }
  for (int x = 0; x < w; ++x)
    {
      diffX[x] = x - cenX;
      diffX2[x] = diffX[x] * diffX[x];
    }

  Image<byte>::const_iterator optr = itsObjectMask.begin();
  for (int y = 0; y < h; ++y)
    for(int x = 0; x < w; ++x)
      {
        if (*optr != 0)
          {
            itsUxx += diffX2[x];
            itsUyy += diffY2[y];
            itsUxy += (diffX[x] * diffY[y]);
          }
        ++optr;
      }
  itsUxx /= itsArea;
  itsUyy /= itsArea;
  itsUxy /= itsArea;

  // compute the parameters d, e and f for the ellipse:
  // d*x^2 + 2*e*x*y + f*y^2 <= 1
  float coeff = 1 / (4 * (itsUxx * itsUyy - itsUxy * itsUxy));
  float d =  coeff * itsUyy;
  float e = -coeff * itsUxy;
  float f =  coeff * itsUxx;

  // from these guys, compute the paramters d and f for the
  // ellipse when it is rotated so that x is the main axis
  // and figure out the angle of rotation for this.
  float expr = sqrt(4*e*e + squareOf(d - f));
  float d2 = 0.5 * (d + f + expr);
  float f2 = 0.5 * (d + f - expr);

  // the angle is defined in clockwise (image) coordinates:
  //  --  is 0
  //  \   is 45
  //  |   is 90
  //  /   is 135
  itsOriAngle = 90 * atan(2 * e / (d - f)) / M_PI;
  if (itsUyy > itsUxx) itsOriAngle += 90.0F;
  if (itsOriAngle < 0.0F) itsOriAngle += 180.0F;

  // this checks if itsOriAngle is nan
  if (itsOriAngle != itsOriAngle) itsOriAngle = 0.0F;

  // now get the length of the major and the minor axes:
  itsMajorAxis = 2 / sqrt(f2);
  itsMinorAxis = 2 / sqrt(d2);
  itsElongation = itsMajorAxis / itsMinorAxis;

  // We're done
  haveSecondMoments = true;
  return;
}

// ######################################################################
void BitObject::freeMem()
{
  itsObjectMask.freeMem();
  itsBoundingBox = Rectangle();
  itsCentroidXY = Vector2D();
  itsArea = 0;
  itsUxx = 0.0F;
  itsUyy = 0.0F;
  itsUxy = 0.0F;
  itsMajorAxis = 0.0F;
  itsMinorAxis = 0.0F;
  itsElongation = 0.0F;
  itsOriAngle = 0.0F;
  itsImageDims = Dims(0,0);
  itsMaxIntensity = -1.0F;
  itsMinIntensity = -1.0F;
  itsAvgIntensity = -1.0F;
  haveSecondMoments = false;
}
// ######################################################################
void BitObject::writeToStream(std::ostream& os) const
{
  // bounding box
  if (itsBoundingBox.isValid())
    {
      os << itsBoundingBox.top() << " "
         << itsBoundingBox.left() << " "
         << itsBoundingBox.bottomI() << " "
         << itsBoundingBox.rightI() << "\n";
    }
  else
    {
      os << "-1 -1 -1 -1\n";
    }

  // image dimensions
  os << itsImageDims.w() << " " << itsImageDims.h() << "\n";

  // centroid
  itsCentroidXY.writeToStream(os);

  // area
  os << itsArea << "\n";

  // have second moments?
  if (haveSecondMoments) os << "0\n";
  else os <<"1\n";

  // second moments
  os << itsUxx << " " << itsUyy << " " << itsUxy << "\n";

  // axes, elongation, angle
  os << itsMajorAxis << " "
     << itsMinorAxis << " "
     << itsElongation << " "
     << itsOriAngle << "\n";

  // max, min and avg intensity
  os << itsMaxIntensity << " "
     << itsMinIntensity << " "
     << itsAvgIntensity << "\n";

  // the object mask
  PnmWriter::writeAsciiBW(itsObjectMask, 1, os);

  os << "\n";

  // done
  return;
}

// ######################################################################
void BitObject::readFromStream(std::istream& is)
{
  // bounding box
  int t, l, b, r;
  is >> t; is >> l; is >> b; is >> r;
  if (t >= 0)
    itsBoundingBox = Rectangle::tlbrI(t, l, b, r);
  else
    itsBoundingBox = Rectangle();


  // image dims
  int w, h;
  is >> w; is >> h;
  itsImageDims = Dims(w,h);

  // centroid
  itsCentroidXY = Vector2D(is);

  // area
  is >> itsArea;

  // have second moments?
  int hs; is >> hs;
  haveSecondMoments = (hs == 1);

  // second moments
  is >> itsUxx; is >> itsUyy; is >> itsUxy;

  // axes, elongation, angle
  is >> itsMajorAxis; is >> itsMinorAxis;
  is >> itsElongation; is >> itsOriAngle;

  // max, min, avg intensity
  is >> itsMaxIntensity;
  is >> itsMinIntensity;
  is >> itsAvgIntensity;

  // object mask
  PnmParser pp(is);
  itsObjectMask = pp.getFrame().asGray();
}

// ######################################################################
template <class T>
void BitObject::setMaxMinAvgIntensity(const Image<T>& img)
{
  ASSERT(img.getDims() == itsImageDims);
  if (!isValid()) return;

  float sum = 0.0F;
  int num = 0;

  // loop over bounding box
  const int iw = img.getWidth();
  typename Image<byte>::const_iterator bptr = itsObjectMask.begin();
  typename Image<T>::const_iterator iptr2, iptr = img.begin();
  iptr += (iw * itsBoundingBox.top() + itsBoundingBox.left());

  for (int y = itsBoundingBox.top(); y < itsBoundingBox.bottomO(); ++y)
    {
      iptr2 = iptr;
      for (int x = itsBoundingBox.left(); x < itsBoundingBox.rightO(); ++x)
        {
          // check if we're indeed inside the object
          if (*bptr > byte(0))
            {
              sum += (float)(*iptr2);
              ++num;
              if ((itsMaxIntensity == -1.0F) || (*iptr2 > itsMaxIntensity))
                itsMaxIntensity = *iptr2;
              if ((itsMinIntensity == -1.0F) || (*iptr2 < itsMinIntensity))
                itsMinIntensity = *iptr2;
            }
          ++bptr;
          ++iptr2;
        }
      iptr += iw;
    }

  if (sum == 0) itsAvgIntensity = 0.0F;
  else itsAvgIntensity = sum / (float)num;

}

// ######################################################################
void BitObject::getMaxMinAvgIntensity(float& maxIntensity,
                                      float& minIntensity,
                                      float& avgIntensity)
{
  maxIntensity = itsMaxIntensity;
  minIntensity = itsMinIntensity;
  avgIntensity = itsAvgIntensity;
}

// ######################################################################
Rectangle BitObject::getBoundingBox(const BitObject::Coords coords) const
{
  switch(coords)
    {
    case OBJECT: return Rectangle::tlbrI(0, 0, itsBoundingBox.height() - 1,
                                  itsBoundingBox.width() - 1);
    case IMAGE: return itsBoundingBox;
    default: LFATAL("Unknown Coords type - don't know what to do.");
    }
  //this is never reached but we have to make the compiler happy
  return Rectangle();
}

// ######################################################################
Image<byte> BitObject::getObjectMask(const byte value,
                                     const BitObject::Coords coords) const
{
  ASSERT(isValid());
  Image<byte> objectCopy = replaceVals(itsObjectMask,byte(1),value);

  switch (coords)
    {
    case OBJECT: return objectCopy;

    case IMAGE:
      {
        Image<byte> result(itsImageDims, ZEROS);
        pasteImage(result, objectCopy, byte(0), getObjectOrigin());
        return result;
      }

    default: LFATAL("Unknown Coords type - don't know what to do.");
    }

  //this is never reached but we have to make the compiler happy
  return Image<byte>();
}

// ######################################################################
Dims BitObject::getObjectDims() const
{ return itsObjectMask.getDims(); }

// ######################################################################
Point2D<int> BitObject::getObjectOrigin() const
{
  return Point2D<int>(itsBoundingBox.left(),itsBoundingBox.top());
}

// ######################################################################
Point2D<int> BitObject::getCentroid(const BitObject::Coords coords) const
{
  return getCentroidXY().getPoint2D();
}

// ######################################################################
Vector2D BitObject::getCentroidXY(const BitObject::Coords coords) const
{
  switch (coords)
    {
    case OBJECT: return (itsCentroidXY - Vector2D(getObjectOrigin()));
    case IMAGE: return itsCentroidXY;
    default: LFATAL("Unknown Coords type - don't know what to do.");
    }
  //this is never reached but we have to make the compiler happy
  return Vector2D();
}

// ######################################################################
int BitObject::getArea() const
{ return itsArea; }

// ######################################################################
void BitObject::getSecondMoments(float& uxx, float& uyy, float& uxy)
{
  if (!haveSecondMoments) computeSecondMoments();
  uxx = itsUxx; uyy = itsUyy; uxy = itsUxy;
}

// ######################################################################
float BitObject::getMajorAxis()
{
  if (!haveSecondMoments) computeSecondMoments();
  return itsMajorAxis;
}

// ######################################################################
float BitObject::getMinorAxis()
{
  if (!haveSecondMoments) computeSecondMoments();
  return itsMinorAxis;
}

// ######################################################################
float BitObject::getElongation()
{
  if (!haveSecondMoments) computeSecondMoments();
  return itsElongation;
}

// ######################################################################
float BitObject::getOriAngle()
{
  return itsOriAngle;
}

// ######################################################################
bool BitObject::isValid() const
{ return ((itsArea > 0) && itsBoundingBox.isValid()); }

// ######################################################################
bool BitObject::doesIntersect(const BitObject& other) const
{
  // are this and other actually valid? no -> return false
  if (!(isValid() && other.isValid()))
    {
      LINFO("no interesect, because one of the objects is invalid.");
      return false;
    }

  Rectangle tBB = getBoundingBox(IMAGE);
  Rectangle oBB = other.getBoundingBox(IMAGE);

  // compute the intersecting bounding box params
  int ll = std::max(tBB.left(),oBB.left());
  int rr = std::min(tBB.rightI(),oBB.rightI());
  int tt = std::max(tBB.top(),oBB.top());
  int bb = std::min(tBB.bottomI(),oBB.bottomI());

  //LINFO("this.ObjMask.dims = %s; other.ObjMask.dims = %s",
  //    toStr(getObjectMask(byte(1),OBJECT).getDims()).c_str(),
  //    toStr(other.getObjectMask(byte(1),OBJECT).getDims()).c_str());

  // is this a valid rectangle?
  if ((ll > rr)||(tt > bb))
    {
      //LINFO("No intersect because the bounding boxes don't overlap: %s and %s",
      //    toStr(tBB).c_str(),toStr(oBB).c_str());
      return false;
    }

  // get the rectangles in order to crop the object masks
  Rectangle tCM = Rectangle::tlbrI(tt - tBB.top(), ll - tBB.left(),
                                  bb - tBB.top(), rr - tBB.left());
  Rectangle oCM = Rectangle::tlbrI(tt - oBB.top(), ll - oBB.left(),
                                  bb - oBB.top(), rr - oBB.left());

  //LINFO("tCM = %s; oCM = %s; this.ObjMask.dims = %s; other.ObjMask.dims = %s",
  //    toStr(tCM).c_str(),toStr(oCM).c_str(),
  //    toStr(getObjectMask(byte(1),OBJECT).getDims()).c_str(),
  //    toStr(other.getObjectMask(byte(1),OBJECT).getDims()).c_str());
  //LINFO("tCM: P2D = %s, dims = %s; oCM: P2D = %s, dims = %s",
  //    toStr(Point2D<int>(tCM.left(), tCM.top())).c_str(),
  //    toStr(Dims(tCM.width(), tCM.height())).c_str(),
  //    toStr(Point2D<int>(oCM.left(), oCM.top())).c_str(),
  //    toStr(Dims(oCM.width(), oCM.height())).c_str());

  // crop the object masks and get the intersecting image
  Image<byte> cor = takeMin(crop(getObjectMask(byte(1),OBJECT),tCM),
                            crop(other.getObjectMask(byte(1),OBJECT),oCM));
  double s = sum(cor);

  //LINFO("tCM = %s; oCM = %s; this.ObjMask.dims = %s; other.ObjMask.dims = %s; sum = %g",
  //    toStr(tCM).c_str(),toStr(oCM).c_str(),
  //    toStr(getObjectMask(byte(1),OBJECT).getDims()).c_str(),
  //    toStr(other.getObjectMask(byte(1),OBJECT).getDims()).c_str(),s);

  return (s > 0.0);
}

// ######################################################################
template <class T_or_RGB>
void BitObject::drawShape(Image<T_or_RGB>& img,
                          const T_or_RGB& color,
                          float opacity)
{
  ASSERT(isValid());
  ASSERT(img.initialized());
  ASSERT(img.getDims() == itsImageDims);

  int w = img.getWidth();
  float op2 = 1.0F - opacity;

  typename Image<T_or_RGB>::iterator iptr, iptr2;
  Image<byte>::const_iterator mptr = itsObjectMask.begin();
  iptr2 = img.beginw() + itsBoundingBox.top() * w + itsBoundingBox.left();
  for (int y = itsBoundingBox.top(); y < itsBoundingBox.bottomO(); ++y)
    {
      iptr = iptr2;
      for (int x = itsBoundingBox.left(); x < itsBoundingBox.rightO(); ++x)
        {
          if (*mptr > 0) *iptr = T_or_RGB(*iptr * op2 + color * opacity);
          ++iptr; ++mptr;
        }
      iptr2 += w;
    }
}

// ######################################################################
template <class T_or_RGB>
void BitObject::drawOutline(Image<T_or_RGB>& img,
                            const T_or_RGB& color,
                            float opacity)
{
  ASSERT(isValid());
  ASSERT(img.initialized());
  ASSERT(img.getDims() == itsImageDims);
  float op2 = 1.0F - opacity;

  Image<byte> marked(img.getDims(), ZEROS);

  int t = itsBoundingBox.top();
  int b = itsBoundingBox.bottomO();
  int l = itsBoundingBox.left();
  int r = itsBoundingBox.rightO();

  for (int y = t; y < b; ++y)
    for (int x = l; x < r; ++x)
      {
        if (itsObjectMask.getVal(x-l,y-t) == 0) continue;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            {
              if ((dy == 0) && (dx == 0)) continue;
              Point2D<int> pim(x+dx,y+dy);
              if (!img.coordsOk(pim)) continue;

              bool isBoundary = false;
              Point2D<int> pm(x+dx-l,y+dy-t);
              if (!itsObjectMask.coordsOk(pm)) isBoundary = true;
              else if (itsObjectMask.getVal(pm) == 0) isBoundary = true;

              if (isBoundary && marked.getVal(pim) == 0)
                {
                  img.setVal(pim,T_or_RGB(img.getVal(pim) * op2 + color * opacity));
                  marked.setVal(pim,byte(1));
                }
            } // end for dx,dy
      } // end for x,y
} // end drawOutline


// ######################################################################
template <class T_or_RGB>
void BitObject::drawBoundingBox(Image<T_or_RGB>& img,
                                const T_or_RGB& color,
                                float opacity)
{
  ASSERT(isValid());
  ASSERT(img.initialized());
  ASSERT(img.getDims() == itsImageDims);

  float op2 = 1.0F - opacity;
  int t = itsBoundingBox.top();
  int b = itsBoundingBox.bottomI();
  int l = itsBoundingBox.left();
  int r = itsBoundingBox.rightI();

  for (int x = l; x <= r; ++x)
    {
      Point2D<int> p1(x,t), p2(x,b);
      img.setVal(p1,img.getVal(p1) * op2 + color * opacity);
      img.setVal(p2,img.getVal(p2) * op2 + color * opacity);
    }
  for (int y = t+1; y < b; ++y)
    {
      Point2D<int> p1(l,y), p2(r,y);
      img.setVal(p1,img.getVal(p1) * op2 + color * opacity);
      img.setVal(p2,img.getVal(p2) * op2 + color * opacity);
    }
}

// ######################################################################
template <class T_or_RGB>
void BitObject::draw(BitObjectDrawMode mode, Image<T_or_RGB>& img,
                     const T_or_RGB& color, float opacity)
{
  switch(mode)
    {
    case BODMnone:    break;
    case BODMshape:   drawShape(img, color, opacity); break;
    case BODMoutline: drawOutline(img, color, opacity); break;
    case BODMbbox:    drawBoundingBox(img, color, opacity); break;
    default: LERROR("Unknown BitObjectDrawMode: %i - ignoring!",mode);
    }
}

// ######################################################################
// Instantiation of template functions
// ######################################################################
template void BitObject::setMaxMinAvgIntensity(const Image<byte>& img);
template void BitObject::setMaxMinAvgIntensity(const Image<float>& img);

#define INSTANTIATE(T_or_RGB) \
template void BitObject::drawShape(Image< T_or_RGB >& img, \
                                   const T_or_RGB& color, \
                                   float opacity); \
template void BitObject::drawOutline(Image< T_or_RGB >& img, \
                                     const T_or_RGB& color, \
                                     float opacity); \
template void BitObject::drawBoundingBox(Image< T_or_RGB >& img, \
                                         const T_or_RGB& color, \
                                         float opacity); \
template void BitObject::draw(BitObjectDrawMode mode, \
                              Image< T_or_RGB >& img, \
                              const T_or_RGB& color, \
                              float opacity);

INSTANTIATE(PixRGB<byte>);
INSTANTIATE(PixRGB<float>);
INSTANTIATE(byte);
INSTANTIATE(float);

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
