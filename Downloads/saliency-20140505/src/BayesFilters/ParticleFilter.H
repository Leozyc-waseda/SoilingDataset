/*!@file BayesFilters/ParticleFilter.H particle Filter               */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: $
// $Id: $
//

#ifndef ParticleFilter_H_DEFINED
#define ParticleFilter_H_DEFINED

#include "Image/Image.H"
#include "Image/MatrixOps.H"
#include "Image/lapack.H"
#include <stdio.h>
#include <stdlib.h>


class ParticleFilter
{

  public:
    struct Particle
    {
      Image<double> state;
      double weight;
      double cumlProb; //Used for selecting particles
    };

    ParticleFilter(int numStates, int numObservations, int numParticles);
    virtual ~ParticleFilter() {};

    //! The function to move from one state to another (need to be implemented)
    virtual Image<double> getNextState(const Image<double>& X) = 0;

    //! The function to predict the observation from the current state
    virtual Image<double> getObservation(const Image<double>& X) = 0;

    //! Predict the next state and covariance
    void predictState();

    int pickParticleToSample(void);
 
    double getLikelihood(const Image<double>& z, const Image<double>& X);
    

    //! Predict the observations
    void predictObservation();

    //! Update the state and covariance given the observation z
    void update(const Image<double>& z);


    double gaussianRand(void)
    {
      static int next_gaussian = 0;
      static double saved_gaussian_value;

      double fac, rsq, v1, v2;

      if (next_gaussian == 0) {
        do {
          v1 = 2.0*uniformRandom()-1.0;
          v2 = 2.0*uniformRandom()-1.0;
          rsq = v1*v1+v2*v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0*log(rsq)/rsq);
        saved_gaussian_value=v1*fac;
        next_gaussian=1;
        return v2*fac;
      } else {
        next_gaussian=0;
        return saved_gaussian_value;
      }
    }

    double uniformRandom(void)
    {
      return (double) rand() / (double) RAND_MAX;
    }
    
    
    

  protected:
    int itsNumStates;
    int itsNumObservations;

    std::vector<Particle> itsParticles;

  private:
    double largestCumlProb;
    

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif 
