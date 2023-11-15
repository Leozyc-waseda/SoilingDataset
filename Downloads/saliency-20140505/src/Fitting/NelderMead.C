//Nelder-Mead minimization algorithm
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

//primary maintainer: David J. Berg

#ifdef INVT_USE_CPP11//we need c++ 0X features for this to work
#include "Fitting/NelderMead.H"
#include <random>
#include <chrono>
#include <limits>
#include <cmath>
#include <algorithm>
#include <future>

// ######################################################################
// implementation of NelderMead
// ######################################################################
NelderMead::NelderMead(std::vector<double> const & start, 
                       std::vector<double> const & parammin, std::vector<double> const & parammax,
                       double const & expansion, 
                       std::function<double const (std::vector<double> const &)> func, 
                       double const & errortol, uint const maxiter, double const & deltaerrortolerance, 
                       bool const multithread, bool const display)
    : itsErrorTolerance(errortol), itsDeltaErrorTolerance(deltaerrortolerance), 
      itsMaxIter(maxiter), itsMultiThread(multithread), 
      itsDisplay(display), itsFunc(func), itsSimplex(start.size()+1), 
      itsRange()
{ 

  if ((start.size() != parammin.size()) || (start.size() != parammax.size()))
  {
    std::cout << "Starting parameters, parameter minimums and parameter maximums must all be vectors of the same length" << std::endl;
    exit(0);
  }

  auto time = std::chrono::system_clock::now();
  auto since_epoch = time.time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch);
  int seed = millis.count();
  std::mt19937 engine(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto generator = std::bind(distribution, engine);
 
  //one point is our starting point, the rest are randomly perturbed 
  std::vector<VertexPair>::iterator point(itsSimplex.begin()), end(itsSimplex.end());
  point->first = start;
  ++point;
  
  while (point != end)
  {
    if (itsDisplay)
      std::cout << "Initial vertex: " << std::endl;
    
    point->first = start;
    point->second = std::numeric_limits<double>::max();
    
    std::vector<double>::iterator dim = point->first.begin();
    std::vector<double>::const_iterator pmin(parammin.begin()), pmax(parammax.begin());
    while (dim != point->first.end())
    {
      double const minrange = (*pmin++ - *dim) * expansion;
      double const maxrange = (*pmax++ - *dim) * expansion;
      *dim += minrange + generator() * (maxrange - minrange);

      if (itsDisplay)
        std::cout << *dim << " ";
      
      ++dim;
    }
    
    if (itsDisplay)
      std::cout << std::endl;
    
    ++point;
  }

  //store the range of each param
  std::vector<double>::const_iterator pmin(parammin.begin()), pmax(parammax.begin());
  while (pmin != parammin.end())
    itsRange.push_back(*pmax++ - *pmin++);
}

// ######################################################################
NelderMead::Result NelderMead::minimize()
{
  //setup some counters and constants
  uint iterations = 0;
  uint const size = itsSimplex.size();
  bool deltaexit = false;
  
  if (itsMultiThread)
  {
    //Evaluate all the verticies of the simplex using multiple threads. We assume here that the itsFunc is a copyable
    //class and that a deep copy of any data needed to the fitting is made. Very undesirable results will accur of the
    //former is true but not the latter.
    std::vector<VertexPair>::iterator point(itsSimplex.begin()), pend(itsSimplex.end());
    std::vector<std::thread> threads;

    //create a function to do the work
    auto func = [](std::function<double const (std::vector<double> const &)> ff, 
                   std::vector<VertexPair>::iterator pp) { pp->second = ff(pp->first); };

    while (point != pend)
    {
      std::thread thread(func, itsFunc, point);
      threads.push_back(std::move(thread));
      ++point;
    }

    std::vector<std::thread>::iterator thread(threads.begin()), tend(threads.end());
    while (thread != tend)
      (thread++)->join();

    point = itsSimplex.begin();
    while (point != pend)
    { 
      if (itsDisplay)
        std::cout << "iteration: " << iterations << " error: " << point->second << std::endl;
      
      ++point;
    }
  }
  else
  {
    //evaluate all the verticies of the simplex using a single thread
    std::vector<VertexPair>::iterator point(itsSimplex.begin()), end(itsSimplex.end());
    while (point != end)
    {
      point->second = itsFunc(point->first);      

      if (itsDisplay)
        std::cout << "iteration: " << iterations << " error: " << point->second << std::endl;
      
      ++point;
    }
  }

  //run until we hit the max number of iterations
  while (iterations < itsMaxIter)
  {
    ++iterations;
    
    //step 1 : order the points
    
    //reorder the points on the simplex by there evaluation score
    std::sort(itsSimplex.begin(), itsSimplex.end(), 
              [](const VertexPair& a, const VertexPair& b) { return a.second < b.second; });
    
    if (itsDisplay)
    {
      std::cout << "iteration: " << iterations << " error: " << itsSimplex.front().second << std::endl;
      std::cout << "params: ";
      for (double const & p : itsSimplex.front().first)
        std::cout << p << " ";
      std::cout << std::endl;
    }

    //break if we are close enough to our target
    if (itsSimplex.front().second < itsErrorTolerance)
      break;

    //break if our parameters aren't changing much
    std::vector<double>::const_iterator op(itsSimplex.back().first.begin()), np(itsSimplex.front().first.begin()), r(itsRange.begin());
    bool finished = true;
    while (op != itsSimplex.back().first.end())
    {
      double const e = std::abs(*np++ - *op++) / *r++;
      if (e >= itsDeltaErrorTolerance)
        finished = false;
    }

    if (finished)
    {
      deltaexit = true;
      break;
    }
      
    //step 2 : compute centroid of all but worst
    
    std::vector<double> worst = itsSimplex.back().first;
    std::vector<double> centroid(worst.size(), 0.0);
    std::vector<VertexPair>::iterator point(itsSimplex.begin()), end(itsSimplex.end()-1);
    while (point != end)
    {
      std::transform(centroid.begin(), centroid.end(), point->first.begin(), centroid.begin(), std::plus<double>());
      ++point;
    }
    std::transform(centroid.begin(), centroid.end(), centroid.begin(), std::bind2nd(std::divides<double>(), size-1));
    
    //step 3 : reflection
    
    //get reflected point
    VertexPair reflected;
    reflected.first = std::vector<double>(worst.size(), 0.0);
    std::transform(worst.begin(), worst.end(), centroid.begin(), reflected.first.begin(), 
                   [](const double& a, const double& b){ return b + 1*(b - a); });

    //if f(best) <= f(reflect) < f(second worst) replace the worst with the reflected
    reflected.second = itsFunc(reflected.first);
    if ((reflected.second >= itsSimplex.front().second) && (reflected.second < itsSimplex[size-2].second))
    {
      itsSimplex.back() = reflected;
      
      if (itsDisplay)
        std::cout << "reflection" << std::endl;
      continue;
    }

    //step 4 : expansion
    
    //if (f(best) > f(reflected) compute the expanded point
    if (reflected.second < itsSimplex.front().second)
    {
      VertexPair expanded;
      expanded.first = std::vector<double>(worst.size(), 0.0);
      std::transform(reflected.first.begin(), reflected.first.end(), centroid.begin(), expanded.first.begin(), 
                     [](const double& a, const double& b){ return b + 2*(a - b); });
      
      expanded.second = itsFunc(expanded.first);
      if (expanded.second < reflected.second)
      {
        itsSimplex.back() = expanded;

        if (itsDisplay)
          std::cout << "expansion" << std::endl;
      }
      else
      {
        itsSimplex.back() = reflected;

        if (itsDisplay)
          std::cout << "reflection" << std::endl;
      }
      
      continue;
    }
    
    //step 5 : contraction
    
    //we would only get here if f(reflected) >= f(second worst)
    VertexPair contracted;
    contracted.first = std::vector<double>(worst.size(), 0.0);

    //if f(second worst) <= f(refleced) < f(worst) contract outise
    if (reflected.second < itsSimplex.back().second)
    {
      std::transform(reflected.first.begin(), reflected.first.end(), centroid.begin(), contracted.first.begin(), 
                     [](const double& a, const double& b){ return b + 0.5 * (a - b); });
      contracted.second = itsFunc(contracted.first);
      //if f(contracted) <= f(reflected)
      if (contracted.second <= reflected.second)
      {
        itsSimplex.back() = contracted;
        
        if (itsDisplay)
          std::cout << "contraction outside" << std::endl;
        continue;
      }
    }
    else //f(reflected) >= f(worst) contract inside
    {
      std::transform(itsSimplex.back().first.begin(), itsSimplex.back().first.end(), centroid.begin(), contracted.first.begin(), 
                     [](const double& a, const double& b){ return b + 0.5 * (a - b); });
      contracted.second = itsFunc(contracted.first);

      //if f(contracted) < f(worst)
      if (contracted.second < itsSimplex.back().second)
      {
        itsSimplex.back() = contracted;
        
        if (itsDisplay)
          std::cout << "contraction inside" << std::endl;
        continue;
      }
    }

    //step 6 : reduction

    point = itsSimplex.begin() + 1;
    end = itsSimplex.end();
    while (point != end)
    {
      std::transform(itsSimplex.front().first.begin(), itsSimplex.front().first.end(), point->first.begin(), point->first.begin(), 
                     [](const double& a, const double& b){ return a + 0.5 * (b - a); });
      ++point;
    }

    if (itsDisplay)
      std::cout << "reduction" << std::endl;
    
  }//end while iterations

  Result result;
  result.params = itsSimplex.front();
  result.iterations = iterations;
  
  if (iterations == itsMaxIter)
    result.code = FinishCode::MAXITERATIONS;
  
  else if (deltaexit)
    result.code = FinishCode::DELTAERRORTOLERANCE;
  
  else 
    result.code = FinishCode::ERRORTOLERANCE;
  
  return result;
}

// ######################################################################
void NelderMead::findConstraints(double const & delta_error, double const & delta_param, std::vector<double> & min, std::vector<double> & max)
{
  /*
  if (itsSimplex.size() < 1)
  {
    std::cout << "No verticies in the simplex" << std::endl;
    exit(0);
  }
  
  min = (itsSimplex.begin()->first.size(),0.0);
  max = (itsSimplex.begin()->first.size(),0.0);

  std::vector<double> param = itsSimplex.begin()->first;
  double const error_goal = delta_error * itsSimplex.begin()->second;
  
  std::vector<double>::const_iterator p(param.begin()), r(itsRange.begin());
  while (p != param.end)
  {
    double const shift = delta_param * *r++;
    //positive shift
    while ()
    {
    }
    
    //negative shift
    while ()
    {
    }
    
    ++p;
  }
  */
}

// ######################################################################
std::string NelderMead::finishCodeToString(FinishCode const code)
{
  switch (code)
  {
  case FinishCode::MAXITERATIONS:
    return "Max iterations";
    
  case FinishCode::ERRORTOLERANCE:
    return "Error  tolerance";

  case FinishCode::DELTAERRORTOLERANCE:
    return "Delta error tolerance";
    
  default:
    return "None";
  };

}

#endif
