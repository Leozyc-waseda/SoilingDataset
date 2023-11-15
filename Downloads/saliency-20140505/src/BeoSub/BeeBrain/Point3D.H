/*!@file BeoSub/BeeBrain/Point3D.H A basic 3D point class */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/Point3D.H $
// $Id: Point3D.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef POINT3D_H_DEFINED
#define POINT3D_H_DEFINED

#include <cmath>
#include <string> // for string conversions

//! This is a basic class to encode 3D integer coordinates
/*! This is a completely public class whose goal is just to provide a
 shorthand notation for 3D integer coordinates.
 All methods are inlined, so there is no .C file, just a .H file.
*/

class Point3D
{
public:
  //! The default constructor initializes the coordinates to (0,0)
  inline Point3D();

  //! Initialize the Point2D<int> from horizontal & vertical coordinates
  inline Point3D(const int xx, const int yy, const int zz);

  //! += operator
  inline Point3D& operator+=(const Point3D &p);
  //! -= operator
  inline Point3D& operator-=(const Point3D &p);
  //! *= operator
  inline Point3D& operator*=(const Point3D &p);
  //! /= operator
  inline Point3D& operator/=(const Point3D &p);

  //! + operator
  inline Point3D operator+(const Point3D &p) const;
  //! - operator
  inline Point3D operator-(const Point3D &p) const;
  //! * operator
  inline Point3D operator*(const Point3D &p) const;
  //! / operator
  inline Point3D operator/(const Point3D &p) const;

  //! += operator
  inline Point3D& operator+=(const int val);
  //! -= operator
  inline Point3D& operator-=(const int val);
  //! *= operator
  inline Point3D& operator*=(const int val);
  //! /= operator
  inline Point3D& operator/=(const int val);

  //! + operator
  inline Point3D operator+(const int val) const;
  //! - operator
  inline Point3D operator-(const int val) const;
  //! * operator
  inline Point3D operator*(const int val) const;
  //! / operator
  inline Point3D operator/(const int val) const;

  //! test whether x is positive
  inline bool isValidX() const;

  //! test whether y is positive
  inline bool isValidY() const;

  //! test whether z is positive
  inline bool isValidZ() const;

  //! the square euclidean distance from p
  inline float squdist(const Point3D& p) const;

  //! the euclidean distance from p
  inline float distance(const Point3D& p) const;

  //! 3D coordinates
  int x, y, z;
};


//! == operator
inline bool operator==(const Point3D& p1, const Point3D& p2);

//! != operator
inline bool operator!=(const Point3D& p1, const Point3D& p2);

//! Point3D overload: format is "<int>,<int>,<int>"
std::string convertToString(const Point3D& val);

//! Point3D overload: format is "<int>,<int>,<int>"
void convertFromString(const std::string& str, Point3D& val);






// ######################################################################
inline Point3D::Point3D()
{ x = 0; y = 0; z = 0;}

// ######################################################################
inline Point3D::Point3D(const int xx, const int yy, const int zz)
{ x = xx; y = yy; z = zz;}

// ######################################################################
inline Point3D& Point3D::operator+=(const Point3D &p)
{ x += p.x; y += p.y; z += p.z; return *this; }

// ######################################################################
inline Point3D& Point3D::operator-=(const Point3D &p)
{ x -= p.x; y -= p.y; z -= p.z; return *this; }

// ######################################################################
inline Point3D& Point3D::operator*=(const Point3D &p)
{ x *= p.x; y *= p.y;  z *= p.z; return *this; }

// ######################################################################
inline Point3D& Point3D::operator/=(const Point3D &p)
{ x /= p.x; y /= p.y;  z /= p.z; return *this; }

// ######################################################################
inline Point3D Point3D::operator+(const Point3D &p) const
{ return Point3D(x + p.x, y + p.y, z + p.z); }

// ######################################################################
inline Point3D Point3D::operator-(const Point3D &p) const
{ return Point3D(x - p.x, y - p.y, z - p.z); }

// ######################################################################
inline Point3D Point3D::operator*(const Point3D &p) const
{ return Point3D(x * p.x, y * p.y, z * p.z); }

// ######################################################################
inline Point3D Point3D::operator/(const Point3D &p) const
{ return Point3D(x / p.x, y / p.y, z / p.z); }

// ######################################################################
inline Point3D& Point3D::operator+=(const int val)
{ x += val; y += val; z += val; return *this; }

// ######################################################################
inline Point3D& Point3D::operator-=(const int val)
{ x -= val; y -= val; z -= val; return *this; }

// ######################################################################
inline Point3D& Point3D::operator*=(const int val)
{ x *= val; y *= val;  z *= val; return *this; }

// ######################################################################
inline Point3D& Point3D::operator/=(const int val)
{ x /= val; y /= val;  z /= val; return *this; }

// ######################################################################
inline Point3D Point3D::operator+(const int val) const
{ return Point3D(x + val, y + val,  z + val); }

// ######################################################################
inline Point3D Point3D::operator-(const int val) const
{ return Point3D(x - val, y - val, z - val); }

// ######################################################################
inline Point3D Point3D::operator*(const int val) const
{ return Point3D(x * val, y * val, z * val); }

// ######################################################################
inline Point3D Point3D::operator/(const int val) const
{ return Point3D(x / val, y / val, z / val); }

// ######################################################################
inline bool operator==(const Point3D& p1, const Point3D& p2)
{ return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z; }

// ######################################################################
inline bool operator!=(const Point3D& p1, const Point3D& p2)
{ return p1.x != p2.x || p1.y != p2.y  || p1.z != p2.z; }

// ######################################################################
inline bool Point3D::isValidX() const
{ return (x >= 0); }

// ######################################################################
inline bool Point3D::isValidY() const
{ return (y >= 0); }

// ######################################################################
inline bool Point3D::isValidZ() const
{ return (z >= 0); }

// ######################################################################
inline float Point3D::squdist(const Point3D& p) const
{
  int d1 = p.x - x, d2 = p.y - y, d3 = p.z - z;
  return (d1 * d1 + d2 * d2 + d3 * d3);
}

// ######################################################################
inline float Point3D::distance(const Point3D& p) const
{ return sqrt(squdist(p)); }


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
