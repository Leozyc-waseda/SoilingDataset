/*!@file MBARI/Geometry2D.H - classes for geometry in the plane
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/Geometry2D.H $
// $Id: Geometry2D.H 9422 2008-03-11 07:33:57Z rjpeters $
//

#ifndef GEOMETRY2D_H_DEFINED
#define GEOMETRY2D_H_DEFINED

#include "Image/Point2D.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <iosfwd>

// ######################################################################
//! A vector in the 2D plane
class Vector2D
{
public:
  //! default constructor, constructs an invalid Vector2D
  Vector2D();

  //! construct a Vector2D with coordinates x and y
  Vector2D(float x, float y);

  //! construct a Vector2D from a Point2D<int>
  Vector2D(const Point2D<int>& point);


  //! construct a Vector2D from a Point2D<double>
  Vector2D(const Point2D<double>& point);

  //! Construct from the data contained in the input stream is
  Vector2D(std::istream& is);

  // default copy constructor, assignment operator and destructor okay

  //! reset the entries of the vector
  void reset(float x, float y);

  //! reset the entries of the vector from a Point2D<int>
  void reset(const Point2D<int>& point);

  //! write the entire Vector2D to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the Vector2D from the input stream is
  void readFromStream(std::istream& is);

  //! access the x coordinate
  float x() const;

  //! access the y coordinate
  float y() const;

  //! return a Point2D<int> (rounding the coordinates to int)
  Point2D<int> getPoint2D() const;

  //! compute the dot product of this with other
  float dotProduct(const Vector2D& other) const;


  //! compute the cross product of this with other
  float crossProduct(const Vector2D& other) const;

  //! compute the length of the vector
  float length() const;

  //! normalize the vector to length 1 and return its old length
  float normalize();

  //! compute the Euclildean distance of this from other
  float distance(const Vector2D& other) const;

  //! compute the angle (in degrees) between this and other
  float angle(const Vector2D& other) const;

  //! determine whether this and other are collinear
  bool isCollinear(const Vector2D& other) const;

  //! determine whether this and and other are orthogonal
  bool isOrthogonal(const Vector2D& other) const;

  //! whether this vector is zero
  bool isZero() const;

  //! whether this vector is valid
  bool isValid() const;

  //! dot product
  float operator*(const Vector2D& v) const;

  //! component-based addition
  Vector2D operator+(const Vector2D& v) const;
  //! component-based subtraction
  Vector2D operator-(const Vector2D& v) const;

  //! component-based addition
  Vector2D& operator+=(const Vector2D& v);
  //conponent-based subtraction
  Vector2D& operator-=(const Vector2D& v);

  //! scalar addition
  Vector2D operator+(const float f) const;
  //! scalar subtraction
  Vector2D operator-(const float f) const;
  //! scalar multiplication
  Vector2D operator*(const float f) const;
  //! scalar division
  Vector2D operator/(const float f) const;

  //! scalar addition
  Vector2D& operator+=(const float f);
  //! scalar subtraction
  Vector2D& operator-=(const float f);
  //! scalar multiplication
  Vector2D& operator*=(const float f);
  //! scalar division
  Vector2D& operator/=(const float f);

private:
  float itsX, itsY;
  bool valid;
};

//! == operator
bool operator==(const Vector2D& v1, const Vector2D& v2);
//! != operator
bool operator!=(const Vector2D& v1, const Vector2D& v2);

// ######################################################################
//! A straight line in the 2D plane
class StraightLine2D
{
public:
  //! default constructo, constructs an invalid object
  StraightLine2D();

  //! construct from two Vector2D
  /*!@param point vector determining a point on the line
    @param direction vector determing the direction if the line*/
  StraightLine2D(const Vector2D& point, const Vector2D& direction);

  //! Construct from the data contained in the input stream is
  StraightLine2D(std::istream& is);

  // default copy constructor, assignment operator and destructor okay

  //! reset the straight line
  /*!@param point vector determining a point on the line
    @param direction vector determing the direction if the line*/
  void reset(const Vector2D& point, const Vector2D& direction);

  //! write the entire StraightLine2D to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the StraightLine2D from the input stream is
  void readFromStream(std::istream& is);

  //! get a point on the line
  /*!@param n get point point + n * direction; for n = 0 (default),
    this is the point entered in the constructor or in reset*/
  Vector2D point(float n = 0.0F) const;

  //! returns the direction vector for this straight line
  Vector2D direction() const;

  //! returns a vector to the intersection point of this and other
  /*! if the lines do not intersect (i.e. they are parallel), an
    invalid Vector2D is returned.
    @param n in this variable, the scalar is returned, for which the
    intersection point is this->point(n);
    @param m the scalar, for which the interesection point is
    other.point(m)*/
  Vector2D intersect(const StraightLine2D& other, float&n, float&m) const;

  //! whether this and other are parallel
  bool isParallel(const StraightLine2D& other) const;

  //! whether this and other are orthogonal
  bool isOrthogonal(const StraightLine2D& other) const;

  //! whether pt is on the straight line
  bool isPointOnLine(const Vector2D& pt) const;

  //! whether this and other describe identical straight lines
  bool isIdentical(const StraightLine2D& other) const;

  //! whether this object is valid
  bool isValid() const;

private:
  Vector2D itsPoint, itsDir;
  bool valid;
};



// ######################################################################
//! A segment of a straight line in the 2D plane
class LineSegment2D
{
public:


  //! default constructor, constructs an invalid object
  LineSegment2D();

  //! construct from two Point2D<int>
  /*!@param p1 an endpoint on the line
    @param p2 the second endpoint of the line line*/
  LineSegment2D(const Point2D<int>& p1, const Point2D<int>& p2);

  //! Construct from the data contained in the input stream is
  LineSegment2D(std::istream& is);

  // default copy constructor, assignment operator and destructor okay

  //! reset the straight line
  /*!@param point vector determining a point on the line
    @param direction vector determing the direction if the line*/
  void reset(const Point2D<int>& point, const Point2D<int>& direction);

  //! write the entire StraightLine2D to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the StraightLine2D from the input stream is
  void readFromStream(std::istream& is);

  //! get a point on the line
  /*!@param n get point point + n * direction; for n = 0 (default),
    this is the point entered in the constructor or in reset*/
  Point2D<int> point1() const;

  //! returns the direction vector for this straight line
  Point2D<int> point2() const;

  //!returns the angle of the line
  double angle() const;

  double angleBetween(LineSegment2D &line) const;

  //! returns true if the two line segments intersect.
  //! x and y are set to be the point of intersection of the lines,
  //! whether or not the segments themselves intersect.
  bool intersects(LineSegment2D &line, double &xcoord, double &ycoord) const;

  double distance(Point2D<double> point) const;

  double distance(Point2D<int> point) const { return distance(Point2D<double>(point)); }

  float length() const;

 //! whether this object is valid
  bool isValid() const;


private:
  Point2D<int> itsPoint1;
  Point2D<int> itsPoint2;
  bool valid;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
