//Nelder-Mead minimization algorithm test program
//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
#include "Fitting/NelderMead.H"
#include <vector>
#include <cmath>

#ifdef INVT_USE_CPP11//we need c++ 0X features for this to work

// parabola
struct Parabola
{
    double const operator()(std::vector<double> const & x)
    {
      return (x[0] - 1.0) * (x[0] - 1.0);
    }
};

// Rosenbrocks parabolic valley
struct Rosenbrock
{
    double const operator()(std::vector<double> const & x)
    {
      return 100.0 * pow( (x[1] - x[0] * x[0]), 2.0) + pow( (1.0 - x[0]), 2.0);
    }
};

int main(const int argc, const char **argv)
{
  //minimize parabola
  std::vector<double> start_parabola(1, 10.0), minp(1,0.0), maxp(1 ,100.0);
  Parabola parabola_function;
  NelderMead parabola_simplex(start_parabola, minp, maxp, 1.0, parabola_function, 1e-15, 500, 1e-4, false, true);
  NelderMead::Result final = parabola_simplex.minimize();
  
  NelderMead::VertexPair result = final.params;
  std::vector<double> point = result.first;
  double error = result.second;
  
  auto ii = point.begin();
  std::cout << "params: ";
  while (ii != point.end())
    std::cout << *ii++ << " ";
  std::cout << std::endl;
  std::cout << "stop criteria: "<< parabola_simplex.finishCodeToString(final.code) << ", error: " << error << std::endl;
  
  //minimize rosenbrock function
  std::vector<double> start_rosenbrock(2, 10.0), minr(2, -10.0), maxr(2, 10.0);
  Rosenbrock rosenbrock_function;
  NelderMead rosenbrock_simplex(start_rosenbrock, minr, maxr, 1.0, rosenbrock_function, 1e-15, 5000, 1e-4, false, true);
  final = rosenbrock_simplex.minimize();
  result = final.params;
  point = result.first;
  error = result.second;

  ii = point.begin();
  std::cout << "params: ";
  while (ii != point.end())
    std::cout << *ii++ << " ";
  std::cout << std::endl;
  std::cout << "stop criteria: "<< rosenbrock_simplex.finishCodeToString(final.code) << ", error: " << error << std::endl;

  return 1;
}
#else
int main(const int argc, const char **argv)
{
  return 0;
}
#endif
