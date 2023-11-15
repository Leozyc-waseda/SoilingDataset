/*!@file MBARI/BitObject.H class that holds an object defined by a shape in a
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/BitObject.H $
// $Id: BitObject.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BITOBJECT_H_DEFINED
#define BITOBJECT_H_DEFINED

#include "Image/Image.H"
#include "Image/Rectangle.H"
#include "MBARI/BitObjectDrawModes.H"
#include "MBARI/Geometry2D.H"
#include "Util/Types.H"


//! Object defined by a connected binary pixel region
/*! This class extracts a connected binary pixel region from a
  grayscale image and analyzes a few of its properties*/

class BitObject
{
public:

  //! default Constructor
  BitObject();

  //! Constructor from a byte Image and a location
  /*!@param img Image from which the object is extracted
    @param location a point that is inside the object - the object
    is flooded starting from this seed
    @param threshold for flooding*/
  BitObject(const Image<byte>& img, const Point2D<int> location,
            const byte threshold = 1);

  //! Construct from a byte Image
  /*! @param img image containing only the object
    the object pixels are 1, all other pixels are 0*/
  BitObject(const Image<byte>& img);

  //! Construct from the data contained in the input stream is
  BitObject(std::istream& is);

  // default copy constructor, operator= and destructor okay

  //! Reset to a new object
  /*!@param img Image from which the object is extracted
    @param location a point that is inside the object - the object
    is flooded starting from this seed
    @param threshold for flooding
    @return a mask of the extracted object (in IMAGE coordinates) */
  Image<byte> reset(const Image<byte>& img, const Point2D<int> location,
                    const byte threshold = 1);

  //! Reset to a new object
  /*! @param img image containing only the object
    the object pixels are 1, all other pixels are 0;
    @return the area of the extracted object; -1 if no object could
    be extracted - in this case the BitObject is invalid */
  int reset(const Image<byte>& img);

  //! delete all stored data, makes the object invalid
  void freeMem();

  //! write the entire BitObject to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the BitObject from the input stream is
  void readFromStream(std::istream& is);

  //! Coordinate system for return values
  /*! These values are used to specify whether return values should be
    given in coordinates of the extracted object or in coordinates
    of the image from which the object was extracted.*/
  enum Coords { OBJECT, IMAGE };

  //! Extract the maximum, minimum and average intensity over the object shape in img
  /*! The results are stored in BitObject and can be accessed
    using getMaxMinAvgIntensity*/
  template <class T>
  void setMaxMinAvgIntensity(const Image<T>& img);

  //! Return the maximum, minimum and average intensity
  /*! See setMinMaxAvgIntensity for details*/
  void getMaxMinAvgIntensity(float& maxIntensity, float& minIntensity,
                             float& avgIntensity);

  //! Returns the bounding box of the object
  Rectangle getBoundingBox(const Coords coords = IMAGE) const;

  //! Returns the object as a binary mask
  /*!@param value the value that is used for the object*/
  Image<byte> getObjectMask(const byte value = byte(1),
                            const Coords coords = IMAGE) const;

  //! The dimensions of the bounding box of the object
  Dims getObjectDims() const;

  //! The location of the upper left corner of the boundary box in the original image
  Point2D<int> getObjectOrigin() const;

  //! The centroid of the object rounded to integer coordinates
  Point2D<int> getCentroid(const Coords coords = IMAGE) const;

  //! The centroid, not rounded
  Vector2D getCentroidXY(const Coords coords = IMAGE) const;

  //! The area (number of pixels) of the object
  int getArea() const;

  //! Returns the second moments of the object (normalized by the area)
  void getSecondMoments(float& uxx, float& uyy, float& uxy);

  //! Returns the length of the major axis of the object
  float getMajorAxis();

  //! Returns the length of the minor axis of the object
  float getMinorAxis();

  //! Returns the elongation of the object (major axis / minor axis)
  float getElongation();

  //! Returns the angle between the major axis and the x axis
  /* @return "--" is 0; "\" is 45; "|" is 90; "/" is 135 */
  float getOriAngle();

  //! whether the object is valid
  /*! This is going to be false if no object could be extracted
    from the image that was given at the constructor or at reset */
  bool isValid() const;

  //! returns the intersection of this BitObject with other
  /*! if the intersection is empty, an invalid BitObject is returned*/
  //BitObject intersection(const BitObject& other);
  bool doesIntersect(const BitObject& other) const;

  //! draw the shape of this BitObject into img with color
  template <class T_or_RGB>
  void drawShape(Image<T_or_RGB>&, const T_or_RGB& color,
                 float opacity = 1.0F);

  //! draw the outline of this BitObject into img with color
  template <class T_or_RGB>
  void drawOutline(Image<T_or_RGB>&, const T_or_RGB& color,
                   float opacity = 1.0F);

  //! draw the bounding box of this BitObject into img with color
  template <class T_or_RGB>
  void drawBoundingBox(Image<T_or_RGB>&,
                       const T_or_RGB& color,
float opacity = 1.0F);

  //! draw this BitObject according to mode
  template <class T_or_RGB>
  void draw(BitObjectDrawMode mode,
            Image<T_or_RGB>&,
            const T_or_RGB& color,
            float opacity = 1.0F);

private:
  // compute the second moments and values derived from them
  void computeSecondMoments();

  Image<byte> itsObjectMask;
  Rectangle itsBoundingBox; // in image coordinates
  Vector2D itsCentroidXY; // in image coordinates
  int itsArea;
  float itsUxx, itsUyy, itsUxy;
  float itsMajorAxis, itsMinorAxis, itsElongation, itsOriAngle;
  Dims itsImageDims;
  float itsMaxIntensity, itsMinIntensity, itsAvgIntensity;
  bool haveSecondMoments;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
