/*!@file BayesFilters/UKF.H Unscented Kalman Filter               */

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

#ifndef UKF_H_DEFINED
#define UKF_H_DEFINED

#include "Image/Image.H"
#include "Image/MatrixOps.H"
#include "Image/lapack.H"
#include <stdio.h>
#include <stdlib.h>


class UKF
{
  public:
    UKF(int numStates, int numObservations,
        double k = 0, //secondery scaling param
        double alpha = 1e-3,  //the spread of the sigma points around the mean
        double beta = 2 //Used to incorporate prior knowledge of the distribution
        );

    virtual ~UKF() {};

    //! The function to move from one state to another (need to be implemented)
    virtual Image<double> getNextState(const Image<double>& X, int k) = 0;

    //! The function to predict the observation from the current state
    virtual Image<double> getObservation(const Image<double>& X, int k) = 0;

    //! Predict the next state and covariance
    void predictState(const Image<double>& noise=Image<double>());

    //! Predict the observations
    void predictObservation(const Image<double>& noise=Image<double>());

    //! Update the state and covariance given the observation z
    void update(const Image<double>& z, const Image<double>& noise);
 
    //! Get the likelihood of a mesurment
    double getLikelihood(const Image<double>& z, const Image<double>& observationNoise);

  protected:
    int itsNumStates;
    int itsNumObservations;

    //Scaling factor which determine how far sigma points a spread from the mean
    double itsAlpha;
    double itsK;

    //used to incude high order information about the distribution
    double itsBeta;

    //The state mean and covariance
    Image<double> itsState;
    Image<double> itsSigma;

    //Noise models
    Image<double> itsR; //processes noise
    Image<double> itsQ; //Measurement noise


    Image<double> getSigmaLocations(const Image<double>& state,
        const Image<double>& sigma, double gamma);

    Image<double> itsSigmaLocations; //The locations we are going to sample from
    Image<double> itsNewStates; //The predicted next state from the sigma locations 
    Image<double> itsNewZ; //the predicted observations from the sigma locations 
    Image<double> itsPredictedZ; //the predicted observations
    Image<double> itsPredictedZSigma; //the predicted observations covariance


  private:
    double itsLambda;
    double itsGamma;
    Image<double> itsMuWeight;
    Image<double> itsSigmaWeight;
    double itsGaussNormalizer;
    bool itsUnstable; //is the covariance symmetric?
    

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif 
