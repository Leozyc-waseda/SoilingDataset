/*!@file BayesFilters/ParticleFilter.C Particle Filter               */
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

#ifndef ParticleFilter_C_DEFINED
#define ParticleFilter_C_DEFINED

#include "BayesFilters/ParticleFilter.H"
#include "Util/MathFunctions.H"

// ######################################################################

ParticleFilter::ParticleFilter(int numStates, int numObservations, int numParticles) :
  itsNumStates(numStates),
  itsNumObservations(numObservations)
{

  //Initialize the particles
  itsParticles.resize(numParticles);
  double totalCumlProb = 0;
  for(uint i=0; i<itsParticles.size(); i++)
  {
    double weight = 1.0;
    itsParticles[1].state = Image<double>(1,itsNumStates,NO_INIT);
    itsParticles[1].weight = weight;
    itsParticles[1].cumlProb = i;
    totalCumlProb += weight;
  }
  largestCumlProb = totalCumlProb;

  double priorMean = 0.0;
  double priorSigma = 0.2;

  //Init the state from a Gaussian
  for(uint i=0; i<itsParticles.size(); i++)
    itsParticles[i].state.setVal(0,0, priorMean + priorSigma*gaussianRand());

}

void ParticleFilter::predictState()
{
  for(uint i=0; i<itsParticles.size(); i++)
  {
    int particle = pickParticleToSample();
    Image<double> newState = getNextState(itsParticles[particle].state);
    itsParticles[particle].state = newState;
  }
}


/* This is binary search using cumulative probabilities to pick a base
   sample. The use of this routine makes Condensation O(NlogN) where N
   is the number of samples. It is probably better to pick base
   samples deterministically, since then the algorithm is O(N) and
   probably marginally more efficient, but this routine is kept here
   for conceptual simplicity and because it maps better to the
   published literature. */
int ParticleFilter::pickParticleToSample(void)
{
  double choice = uniformRandom() * largestCumlProb;
  int low, middle, high;

  low = 0;
  high = itsParticles.size();

  while (high>(low+1)) {
    middle = (high+low)/2;
    if (choice > itsParticles[middle].cumlProb)
      low = middle;
    else high = middle;
  }

  return low;
}



double ParticleFilter::getLikelihood(const Image<double>& z, const Image<double>& X)
{

  Image<double> predZ = getObservation(X);
  double val = predZ[0] - z[0];
  double sigma = 0.03;
  
  //Evaluate using a Gaussian distribution
  /* This private definition is used for portability */
  static const double PI = 3.14159265358979323846;

  return 1.0/(sqrt(2.0*PI) * sigma) *
    exp(-0.5 * (val*val / (sigma*sigma)));

}


void ParticleFilter::update(const Image<double>& z)
{

  double cumlTotal = 0;
  for(uint i=0; i<itsParticles.size(); i++)
  {
    double weight = getLikelihood(z, itsParticles[i].state);
    itsParticles[i].weight = weight;
    itsParticles[i].cumlProb = cumlTotal;
    cumlTotal += weight;
  }
  largestCumlProb = cumlTotal;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif 
