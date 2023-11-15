/*!@file MBARI/Geometry2D.C - classes for geometry in the plane
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/Geometry2D.C $
// $Id: Geometry2D.C 9422 2008-03-11 07:33:57Z rjpeters $
//


#include "MBARI/Geometry2D.H"

#include "Image/Point2D.H"
#include "Util/Assert.H"
#include "Util/MathFunctions.H" // for isFinite()
#include <cmath>
#include <istream>
#include <ostream>

// ######################################################################
// ## implementation of Vector2D
// ######################################################################
Vector2D::Vector2D()
  : itsX(0), itsY(0), valid(false)
{}

// ######################################################################
Vector2D::Vector2D(float x, float y)
  : itsX(x), itsY(y), valid(true)
{}

// ######################################################################
Vector2D::Vector2D(const Point2D<int>& point)
  : itsX(point.i), itsY(point.j), valid(true)
{}

// ######################################################################
Vector2D::Vector2D(const Point2D<double>& point)
  : itsX(point.i), itsY(point.j), valid(true)
{}

// ######################################################################
Vector2D::Vector2D(std::istream& is)
{
  readFromStream(is);
}


// ######################################################################
void Vector2D::reset(float x, float y)
{
  itsX = x;
  itsY = y;
  valid = true;
}

// ######################################################################
void Vector2D::reset(const Point2D<int>& point)
{
  itsX = point.i;
  itsY = point.j;
  valid = true;
}

// ######################################################################
void Vector2D::writeToStream(std::ostream& os) const
{
  if (valid) os << "1\n";
  else os << "0\n";
  os << itsX << ' ' << itsY << '\n';
}

// ######################################################################
void Vector2D::readFromStream(std::istream& is)
{
  int i; is >> i;
  valid = (i == 1);
  is >> itsX;
  is >> itsY;
}

// ######################################################################
float Vector2D::x() const
{
  ASSERT(isValid());
  return itsX;
}

// ######################################################################
float Vector2D::y() const
{
  ASSERT(isValid());
  return itsY;
}

// ######################################################################
Point2D<int> Vector2D::getPoint2D() const
{
  ASSERT(isValid());
  return Point2D<int>(int(itsX + 0.5F),int(itsY + 0.5F));
}

// ######################################################################
float Vector2D::dotProduct(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return (itsX * other.x() + itsY * other.y());
}

// ######################################################################
float Vector2D::crossProduct(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());

  return (itsX * other.y() - itsY * other.x());
}


// ######################################################################
float Vector2D::length() const
{ return sqrt(dotProduct(*this)); }

// ######################################################################
float Vector2D::normalize()
{
  ASSERT(isValid());
  float l = length();
  if (l != 0.0F)
    {
      itsX /= l;
      itsY /= l;
    }
  return l;
}

// ######################################################################
float Vector2D::distance(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  Vector2D diff = *this - other;
  return diff.length();
}

// ######################################################################
float Vector2D::angle(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return acos(dotProduct(other)/length()/other.length())*180.0F/M_PI;
}

// ######################################################################
bool Vector2D::isCollinear(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return (fabs(angle(other)) < 0.1F);
}

// ######################################################################
bool Vector2D::isOrthogonal(const Vector2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return (fabs(dotProduct(other)) < 1e-6);
}

// ######################################################################
bool Vector2D::isZero() const
{
  ASSERT(isValid());
  return (fabs(itsX) < 1e-6) && (fabs(itsY) < 1e-6);
}

// ######################################################################
bool Vector2D::isValid() const
{ return valid; }

// ######################################################################
// ## operators for Vector2D
// ######################################################################
float Vector2D::operator*(const Vector2D& v) const
{ return dotProduct(v); }

// ######################################################################
Vector2D Vector2D::operator+(const Vector2D& v) const
{
  if (!isValid() || !v.isValid()) return Vector2D();
  else return Vector2D(itsX + v.x(), itsY + v.y());
}

// ######################################################################
Vector2D Vector2D::operator-(const Vector2D& v) const
{
  if (!isValid() || !v.isValid()) return Vector2D();
  else return Vector2D(itsX - v.x(), itsY - v.y());
}

// ######################################################################
Vector2D& Vector2D::operator+=(const Vector2D& v)
{
  if (!isValid() || !v.isValid()) valid = false;
  else { itsX += v.x(); itsY += v.y(); }
  return *this;
}

// ######################################################################
Vector2D& Vector2D::operator-=(const Vector2D& v)
{
  if (!isValid() || !v.isValid()) valid = false;
  else { itsX -= v.x(); itsY -= v.y(); }
  return *this;
}

// ######################################################################
Vector2D Vector2D::operator+(const float f) const
{
  if (!isValid()) return Vector2D();
  else return Vector2D(itsX + f, itsY + f);
}

// ######################################################################
Vector2D Vector2D::operator-(const float f) const
{
  if (!isValid()) return Vector2D();
  else return Vector2D(itsX - f, itsY - f);
}

// ######################################################################
Vector2D Vector2D::operator*(const float f) const
{
  if (!isValid()) return Vector2D();
  else return Vector2D(itsX * f, itsY * f);
}

// ######################################################################
Vector2D Vector2D::operator/(const float f) const
{
  ASSERT(f != 0.0F);
  if (!isValid()) return Vector2D();
  else return Vector2D(itsX / f, itsY / f);
}


// ######################################################################
Vector2D& Vector2D::operator+=(const float f)
{
  itsX += f; itsY += f;
  return *this;
}

// ######################################################################
Vector2D& Vector2D::operator-=(const float f)
{
  itsX -= f; itsY -= f;
  return *this;
}

// ######################################################################
Vector2D& Vector2D::operator*=(const float f)
{
  itsX *= f; itsY *= f;
  return *this;
}

// ######################################################################
Vector2D& Vector2D::operator/=(const float f)
{
  ASSERT (f != 0.0F);
  itsX /= f; itsY /= f;
  return *this;
}

// ######################################################################
// ## comparison operators for Vector2D
// ######################################################################
bool operator==(const Vector2D& v1, const Vector2D& v2)
{ return (v1.x() == v2.x()) && (v1.y() == v2.y()); }

bool operator!=(const Vector2D& v1, const Vector2D& v2)
{ return (v1.x() != v2.x()) || (v1.y() != v2.y()); }



// ######################################################################
// ### Implementation of StraightLine2D
// ######################################################################
StraightLine2D::StraightLine2D()
  : itsPoint(), itsDir(), valid(false)
{}

// ######################################################################
StraightLine2D::StraightLine2D(const Vector2D& point,
                               const Vector2D& direction)
  : itsPoint(point), itsDir(direction), valid(true)
{}

// ######################################################################
StraightLine2D::StraightLine2D(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void StraightLine2D::reset(const Vector2D& point,
                           const Vector2D& direction)
{
  itsPoint = point;
  itsDir = direction;
  valid = true;
}
// ######################################################################
void StraightLine2D::writeToStream(std::ostream& os) const
{
  if (valid) os << "1\n";
  else os << "0\n";
  itsPoint.writeToStream(os);
  itsDir.writeToStream(os);
}

// ######################################################################
void StraightLine2D::readFromStream(std::istream& is)
{
  int i; is >> i;
  valid = (i == 1);
  itsPoint.readFromStream(is);
  itsDir.readFromStream(is);
}

// ######################################################################
Vector2D StraightLine2D::point(float n) const
{
  ASSERT(isValid());
  return (itsPoint + itsDir * n);
}

// ######################################################################
Vector2D StraightLine2D::direction() const
{
  ASSERT(isValid());
  return itsDir;
}

// ######################################################################
Vector2D StraightLine2D::intersect(const StraightLine2D& other,
                                   float&n, float&m) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  if (isParallel(other)) return Vector2D();

  Vector2D dif = itsPoint - other.point();
  Vector2D od = other.direction();
  float en = (itsDir.y() * od.x() - od.y() * itsDir.x());
  n = (od.y() * dif.x() - od.x() * dif.y()) / en;
  m = (itsDir.y() * dif.x() - itsDir.x() * dif.y()) / en;

  return point(n);
}

// ######################################################################
bool StraightLine2D::isParallel(const StraightLine2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return (itsDir.isCollinear(other.direction()));
}

// ######################################################################
bool StraightLine2D::isOrthogonal(const StraightLine2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());
  return (itsDir.isOrthogonal(other.direction()));
}

// ######################################################################
bool StraightLine2D::isPointOnLine(const Vector2D& pt) const
{
  ASSERT(isValid());
  ASSERT(pt.isValid());

  float n1 = itsDir.y() * (pt.x() - itsPoint.x());
  float n2 = itsDir.x() * (pt.y() - itsPoint.y());

  return (fabs(n1 - n2) < 1e-6);
}

// ######################################################################
bool StraightLine2D::isIdentical(const StraightLine2D& other) const
{
  ASSERT(isValid());
  ASSERT(other.isValid());

  return (isParallel(other) && isPointOnLine(other.point()));
}

// ######################################################################
bool StraightLine2D::isValid() const
{ return valid; }


// ######################################################################
// ### Implementation of LineSegment2D
// ######################################################################
LineSegment2D::LineSegment2D()
  : itsPoint1(), itsPoint2(), valid(false)
{}

// ######################################################################
LineSegment2D::LineSegment2D(const Point2D<int>& p1,
                             const Point2D<int>& p2)
  : itsPoint1(p1), itsPoint2(p2), valid(p1 != p2)
{}

// ######################################################################
LineSegment2D::LineSegment2D(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void LineSegment2D::reset(const Point2D<int>& p1,
                          const Point2D<int>& p2)
{
  itsPoint1 = p1;
  itsPoint2 = p2;

  if (itsPoint1 != itsPoint2) {
    valid = true;
  }
  else {
    valid = false;
  }
}
// ######################################################################
void LineSegment2D::writeToStream(std::ostream& os) const
{
  if (valid) os << "1\n";
  else os << "0\n";
  Vector2D(itsPoint1).writeToStream(os);
  Vector2D(itsPoint2).writeToStream(os);
}

// ######################################################################
void LineSegment2D::readFromStream(std::istream& is)
{
  int i; is >> i;
  valid = (i == 1);
  Vector2D(itsPoint1).readFromStream(is);
  Vector2D(itsPoint2).readFromStream(is);
}

// ######################################################################
Point2D<int> LineSegment2D::point1() const
{
  ASSERT(isValid());
  return (itsPoint1);
}

// ######################################################################
Point2D<int> LineSegment2D::point2() const
{
  ASSERT(isValid());
  return (itsPoint2);
}

// ######################################################################
double LineSegment2D::angle() const
{
  float o = (float)(point1().j - point2().j);
  float a = (float)(point1().i - point2().i);
  return atan(o/a);
}

// ######################################################################
double LineSegment2D::angleBetween(LineSegment2D &line) const
{
  return fabs(angle() - line.angle());
}

// ######################################################################
float LineSegment2D::length() const
{
  return (float)(point1().distance(point2()));
}
// ######################################################################
bool LineSegment2D::isValid() const
{ return valid; }


// ######################################################################
bool LineSegment2D::intersects(LineSegment2D &line, double &xcoord, double &ycoord) const
{

  double Ax = point1().i;
  double Ay = point1().j;
  double Bx = point2().i;
  double By = point2().j;
  double Cx = line.point1().i;
  double Cy = line.point1().j;
  double Dx = line.point2().i;
  double Dy = line.point2().j;


  double  x, y, distAB, theCos, theSin, newX, ABpos ;

  //  Fail if either line is undefined.
  if ((Ax==Bx && Ay==By) || (Cx==Dx && Cy==Dy)) { return false; }

  //  (1) Translate the system so that point A is on the origin.
  Bx-=Ax; By-=Ay;
  Cx-=Ax; Cy-=Ay;
  Dx-=Ax; Dy-=Ay;

  //  Discover the length of segment A-B.
  distAB=sqrt(Bx*Bx+By*By);

  //  (2) Rotate the system so that point B is on the positive X axis.
  theCos=Bx/distAB;
  theSin=By/distAB;
  newX=Cx*theCos+Cy*theSin;
  Cy  =Cy*theCos-Cx*theSin; Cx=newX;
  newX=Dx*theCos+Dy*theSin;
  Dy  =Dy*theCos-Dx*theSin; Dx=newX;

  //  Fail if the lines are parallel.
  if (Cy==Dy) {
    if (distance(line.point1()) && distance(line.point2()) && line.distance(point1()) && line.distance(point2())) {
      return false;
    }
  }

  //  (3) Discover the position of the intersection point along line
  //  A-B.
  ABpos=Dx+(Cx-Dx)*Dy/(Dy-Cy);

  //  (4) Apply the discovered position to line A-B in the original
  //  coordinate system.
  x=Ax+ABpos*theCos;
  y=Ay+ABpos*theSin;

  //Point2D<int> temp(x, y);


  //  Success.

  //in case the two lines share only an endpoint
  if (point1() == line.point1() || point1() == line.point2()) {
    x = point1().i;
    y = point1().j;
  }
  if (point2() == line.point1() || point2() == line.point2()) {
    x = point2().i;
    y = point2().j;
  }

  xcoord = x;
  ycoord = y;

  //the distance from a line to a pointon the line SHOULD be 0,
  //however, it looks as though compounded errors from conversions to
  //float and double give it some leeway.
  if (!isFinite(x) || !isFinite(y) ||
      distance(Point2D<double>(x, y)) > 0.000010 ||
      line.distance(Point2D<double>(x, y)) > 0.000010  ) {

    return false;
  }

  return true;


}

double LineSegment2D::distance(Point2D<double> point) const {
  Vector2D A(point1());
  Vector2D B(point2());

  Vector2D C(point);

  double dist = ((B-A).crossProduct(C-A)) / sqrt((B-A)*(B-A));
  double dot1 = (C-B)*(B-A);
  if(dot1 > 0)return sqrt((B-C)*(B-C));
  double dot2 = (C-A)*(A-B);
  if(dot2 > 0)return sqrt((A-C)*(A-C));
  return fabs(dist);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
