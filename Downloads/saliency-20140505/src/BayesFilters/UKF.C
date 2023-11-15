/*!@file BayesFilters/UKF.C Unscented Kalman Filter               */
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

#ifndef UKF_C_DEFINED
#define UKF_C_DEFINED

#include "BayesFilters/UKF.H"
#include "Util/MathFunctions.H"

// ######################################################################

UKF::UKF(int numStates, int numObservations,
    double k, double alpha, double beta) :
  itsNumStates(numStates),
  itsNumObservations(numObservations),
  itsAlpha(alpha),
  itsK(k),
  itsBeta(beta),
  itsUnstable(false)
{
  itsR=Image<double>(itsNumStates,itsNumStates,ZEROS);
  itsQ=Image<double>(itsNumObservations,itsNumObservations,ZEROS);


  itsLambda = itsAlpha*itsAlpha*(itsNumStates+itsK)-itsNumStates;
  itsGamma = sqrt(itsNumStates + itsLambda);

  //Calculate the weights for the mean and coveriance ahead of time
  itsMuWeight = Image<double>(1+2*itsNumStates,1,ZEROS);
  itsSigmaWeight = Image<double>(1+2*itsNumStates,1,ZEROS);

  //Initial weight for the mean location
  itsMuWeight[0] = itsLambda/(itsNumStates + itsLambda);
  itsSigmaWeight[0] = (itsLambda/(itsNumStates + itsLambda)) +
                       (1-(itsAlpha*itsAlpha)+itsBeta);

  double weight = 1/(2*(itsNumStates+itsLambda));
  for(int i=1; i<1+itsNumStates*2; i++)
  {
    itsMuWeight[i] = weight;
    itsSigmaWeight[i] = weight;
  }

  //Initial state
  itsState = Image<double>(1,itsNumStates,ZEROS);
  itsSigma = Image<double>(itsNumStates,itsNumStates,ZEROS);

  itsGaussNormalizer = 1.0/sqrt(pow(2.0*M_PI,itsNumStates)); 
}

void UKF::predictState(const Image<double>& noise)
{
  if (itsUnstable) return;
  //Get the locations to sample from
  itsSigmaLocations = getSigmaLocations(itsState, itsSigma, itsGamma);

  //Predict the new state by simulating the process at the sigma points
  //and then computing a new mean by a weighted sum
  itsNewStates = Image<double>(itsSigmaLocations.getDims(),ZEROS);
  Image<double> predictedState(1,itsNumStates,ZEROS);
  for(int k=0; k<itsSigmaLocations.getWidth(); k++)
  {
    Image<double> state = getNextState(itsSigmaLocations,k);
    ASSERT(state.size() == (uint)itsNumStates);
    predictedState += state*itsMuWeight[k];

    for(uint i=0; i<state.size(); i++)
      itsNewStates.setVal(k, i, state[i]);
  }

  //Compute the predicted covariance in a similar manner
  Image<double> predictedSigma(itsSigma.getDims(),ZEROS);
  for(int k=0; k<itsNewStates.getWidth(); k++)
  {
    //Calculate the variance between the new states and the predicted State
    //Could be optimized by unrolling the matrix ops
    
    Image<double> diff(1,predictedState.size(), ZEROS);
    for(uint i=0; i<diff.size(); i++)
      diff[i] = itsNewStates.getVal(k,i)-predictedState[i];

    Image<double> variance = matrixMult(diff, transpose(diff));

    predictedSigma += variance*itsSigmaWeight[k];
  }

  if (noise.initialized())
    predictedSigma += noise;
  else
    predictedSigma += itsR;

  itsState = predictedState;
  itsSigma = predictedSigma;
}

void UKF::predictObservation(const Image<double>& noise)
{
  if (itsUnstable) return;
 //Predict the observation by simulating the process at the sigma points
 //and then computing a new mean by a weighted sum
 
  itsNewZ = Image<double>(itsNewStates.getWidth(),
                       itsNumObservations, ZEROS);
  Image<double> predictedZ(1,itsNumObservations, ZEROS);

  //At each predicted state, predict the observations
  for(int k=0; k<itsNewStates.getWidth(); k++)
  {
    Image<double> observation = getObservation(itsNewStates, k);
    ASSERT(observation.size() == (uint)itsNumObservations);
    predictedZ += observation*itsMuWeight[k];
    for(uint i=0; i<observation.size(); i++)
      itsNewZ.setVal(k, i, observation[i]);
  }

  //Predict the measurement covariance
  
  Image<double> predictedZSigma(itsNumObservations, itsNumObservations,ZEROS);
  for(int k=0; k<itsNewStates.getWidth(); k++)
  {
    //Could be optimized by unrolling the matrix ops
    Image<double> diff(1,predictedZ.size(), ZEROS);
    for(uint i=0; i<diff.size(); i++)
      diff[i] = itsNewZ.getVal(k,i)-predictedZ[i];
    Image<double> variance = matrixMult(diff, transpose(diff));
    predictedZSigma += variance*itsSigmaWeight[k];
  }

  if (noise.initialized())
    predictedZSigma += noise;
  else
    predictedZSigma += itsQ;

  itsPredictedZ = predictedZ;
  itsPredictedZSigma = predictedZSigma;
}

double UKF::getLikelihood(const Image<double>& z, const Image<double>& observationNoise)
{
  if (itsUnstable) return 0;

  if (!itsPredictedZSigma.initialized())
    return 0;

  //Make symmetric

  Image<double> sigma = itsPredictedZSigma;
  if (observationNoise.initialized())
    sigma += observationNoise;

  //sigma = sigma+transpose(sigma);
  //sigma = sigma *0.5;
  double normalizer = itsGaussNormalizer* (1/sqrt(lapack::det(&sigma)));

  ////Calculate the innovation
  Image<double>  innov = z - itsPredictedZ;

  try
  {
    Image<double> probMat = matrixMult(transpose(innov), matrixMult(matrixInv(sigma), innov));
    return normalizer*exp(-0.5*probMat.getVal(0,0));
  } catch (...)
  {
    itsUnstable = true;
  }

  return 0;

}


void UKF::update(const Image<double>& z, const Image<double>& noise)
{
  if (itsUnstable) return;
  //Calculate the covariance between the state and the observation
  Image<double> Pxy(itsNumObservations, itsNumStates, ZEROS);
  for(int k=0; k<itsNewStates.getWidth(); k++)
  {
    //TODO: Since we calculated these diff before, we can save the for optimizations
    Image<double> stateDiff(1,itsNumStates,NO_INIT);
    for(uint i=0; i<stateDiff.size(); i++)
      stateDiff[i] = itsNewStates.getVal(k,i) - itsState[i];

    Image<double> zDiff(1,itsNumObservations,NO_INIT);
    for(uint i=0; i<zDiff.size(); i++)
      zDiff[i] = itsNewZ.getVal(k,i) - itsPredictedZ[i];

    Image<double> variance = matrixMult(stateDiff, transpose(zDiff));
    Pxy += variance*itsSigmaWeight[k];
  }

  try
  {
    Image<double> sigma = itsPredictedZSigma;
    //if (noise.initialized()) //Add the noise if available
    //  sigma += noise;

    //Calculate the kalman gain
    Image<double> K = matrixMult(Pxy, matrixInv(sigma));

    //Calculate the innovation
    Image<double>  innov = z - itsPredictedZ;

    //Update the state and covariance
    itsState += matrixMult(K,innov);
    itsSigma -= matrixMult(K,transpose(Pxy));
  } catch (...)
  {
    itsUnstable = true;
  }


}


Image<double> UKF::getSigmaLocations(const Image<double>& state,
    const Image<double>& sigma, double gamma)
{
  //We use the chol decomposition for stability instead of the sqrt
  
  //Make symmetric
  Image<double> A = sigma;
  //A = A+transpose(A);
  //A = A *0.5;
  Image<double> SChol = lapack::dpotrf(&A);
  //Reset the lower symmetry to 0 (matlab seems to do it for the chol function)
  for(int j=0; j<SChol.getHeight(); j++)
    for(int i=0; i<j; i++)
        SChol.setVal(i,j,0);

  A=transpose(SChol)*gamma;

  Image<double> X(state.size()*2 + 1, state.size(),ZEROS);

  for(int j=0; j<X.getHeight(); j++)
    X.setVal(0,j,state[j]);

  for(int i=0; i<A.getWidth(); i++)
    for(int j=0; j<A.getHeight(); j++)
      X.setVal(1+i,j,state[j]+A.getVal(i,j));

  for(int i=0; i<A.getWidth(); i++)
    for(int j=0; j<A.getHeight(); j++)
      X.setVal(1+state.size()+i,j,state[j]-A.getVal(i,j));

  return X;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif 
