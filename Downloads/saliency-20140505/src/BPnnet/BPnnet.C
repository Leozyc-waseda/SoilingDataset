/*!@file BPnnet/BPnnet.C Back Prop Neural Net class */

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
// Primary maintainer for this file: Philip Williams <plw@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/BPnnet.C $
// $Id: BPnnet.C 6095 2006-01-17 01:15:21Z rjpeters $
//

#include "BPnnet/BPnnet.H"

#include "Image/MathOps.H"
#include "Util/Assert.H"
#include "Util/MathFunctions.H"
#include "Util/log.H"

#include <cstdlib>
#include <fstream>
#include <limits>

// ######################################################################
BPnnet::BPnnet(const int numInput, const int numHidden,
               const KnowledgeBase *kb )
{
  numInputUnits = numInput;

  // changing the number of hidden units can affect (improve/worsen)
  //   the net's performance
  numHiddenUnits = numHidden;

  // numOuputUnits = number visual objects in the knowledge base
  itsKb = kb;
  numOutputUnits = itsKb->getSize();

  //LINFO("input: %d units; hidden: %d units; output: %d units",
  //numInputUnits, numHiddenUnits, numOutputUnits);

  // initialize weights:
  weightFromInput.resize(numInputUnits, numHiddenUnits, true);
  weightToOutput.resize(numHiddenUnits, numOutputUnits, true);

  // initialize input & output vectors:
  inputLayer.resize(numInputUnits);
  hiddenLayer.resize(numHiddenUnits);
  outputLayer.resize(numOutputUnits);
}

// ######################################################################
BPnnet::~BPnnet()
{
  weightFromInput.freeMem();
  weightToOutput.freeMem();
}

// ######################################################################
void BPnnet::randomizeWeights(void)
{
  for (int y = 0; y < numHiddenUnits; y++)
    for (int x = 0; x < numInputUnits; x++)
      weightFromInput.setVal(x, y, (randomDouble() - 0.5) * 0.0001);

  for (int y = 0; y < numOutputUnits; y++)
    for (int x = 0; x < numHiddenUnits; x++)
      weightToOutput.setVal(x, y, (randomDouble() - 0.5) * 0.0001);

  normalizeWeights();
}

// ######################################################################
void BPnnet::normalizeWeights(void)
{
  double mi1, ma1, mi2, ma2;
  getMinMax(weightFromInput, mi1, ma1);
  getMinMax(weightToOutput, mi2, ma2);
  //LINFO("Weights: in=[%f .. %f] out=[%f .. %f]", mi1, ma1, mi2, ma2);
  inplaceClamp(weightFromInput, -10000.0, 10000.0);
  inplaceClamp(weightToOutput, -10000.0, 10000.0);
  //  weightFromInput /= sum(abs(weightFromInput));
  //  weightToOutput /= sum(abs(weightToOutput));
}

// ######################################################################
double BPnnet::train(const Image<float> &in, const SimpleVisualObject& target,
                     const double learnRate)
{
  // Check that target exists in knowledge base itsKb
  // and determine its id, unless target name is "unknown"
  //   ie which output neuron should associate with it
  //   (1-to-1 mapping between the VO's array index in the KB
  //     and the neuron's array index in the outputLayer)
  int targetNeuron = itsKb->findSimpleVisualObjectIndex(target.getName());
  if (targetNeuron == -1 && strcmp(target.getName(), "unknown"))
    LFATAL("Cannot train on unknown object '%s'", target.getName());

  // set expected output levels
  double expectedOutput[numOutputUnits];
  for (int i = 0; i < numOutputUnits; i++) expectedOutput[i] = 0.0;
  if (targetNeuron >= 0) expectedOutput[targetNeuron] = 1.0;

  // do a forward pass:
  forwardProp(in);

  double rms1 = 0.0; // intermediate value for RMS error calculation
  // Calculate error for output
  for (int i = 0; i < numOutputUnits; i++)
    {
      double rms0 = outputLayer[i].calcOutputDelta(expectedOutput[i]);
      rms1 += rms0 * rms0;
    }
  rms1 /= numOutputUnits;

  // Back prop errors to hidden layer
  for (int j = 0; j < numHiddenUnits; j++)
    {
      // Calculate weighted sum of the delta values of all units that
      //   receive ouput from hidden unit j
      // [weightedDeltaSum = sum_k(delta_k * w_kj)]
      double weightedDeltaSum = 0.0;
      for (int k = 0; k < numOutputUnits; k++)
        {
          double delta_k = outputLayer[k].getDelta();
          double w_kj = weightToOutput.getVal(j, k);
          weightedDeltaSum += delta_k * w_kj;
        }

      // Calculate hidden layer delta
      hiddenLayer[j].calcHiddenDelta(weightedDeltaSum);
    }

  // Adjust weights going to output layer
  for (int j = 0; j < numOutputUnits; j++)
    for (int i = 0; i < numHiddenUnits; i++)
      {
        // New weight = learning rate * output neuron's error value *
        //   hidden neuron's activation level.
        // [delta_w_ji = eta * delta_j * a_i]
        double delta_j = outputLayer[j].getDelta(); // error level
        double a_i = hiddenLayer[i].getActivationLevel();
        double weightChange = learnRate * delta_j * a_i;

        weightToOutput.setVal(i, j,
                              weightToOutput.getVal(i, j) + weightChange);

      }

  // Adjust weights coming from input layer
  for (int j = 0; j < numHiddenUnits; j++)
    for (int i = 0; i < numInputUnits; i++)
      {
        // New weight = learning rate * hidden neuron's error value *
        //   input neuron's activation level.
        // [delta_w_ji = eta * delta_j * a_i]
        double delta_j = hiddenLayer[j].getDelta(); // error level
        double a_i = inputLayer[i].getActivationLevel();
        double weightChange = learnRate * delta_j * a_i;

        weightFromInput.setVal(i, j,
                               weightFromInput.getVal(i, j) + weightChange);
      }

  return rms1;
  // whoever called train should know how to use rms1 to calculate RMS
  //   error value for the net's performance during this training cycle

}

// ######################################################################
bool BPnnet::recognize(const Image<float> &in, SimpleVisualObject& vo)
{
  // do a forward propagation:
  forwardProp(in);

  // Determine which output neuron (n) has maximum activationLevel
  double maxOutput = - std::numeric_limits<double>::max();
  double meanOutput = 0.0, maxOutput2 = maxOutput;
  int maxNeuron = -1, maxNeuron2 = -1;
  for (int n = 0; n < numOutputUnits; n++)
    {
      double thisOutput = outputLayer[n].getActivationLevel();
      meanOutput += thisOutput;
      if (thisOutput > maxOutput)
        { maxOutput = thisOutput; maxNeuron = n; }
    }
  for (int n = 0; n < numOutputUnits; n++)
    {
      double thisOutput = outputLayer[n].getActivationLevel();
      if (n != maxNeuron && thisOutput > maxOutput2)
        { maxOutput2 = thisOutput; maxNeuron2 = n; }
    }
  meanOutput /= (double)numOutputUnits;
  LINFO("max for '%s' (%.3f), max2 for '%s' (%.3f), mean=%.3f",
        itsKb->getSimpleVisualObject(maxNeuron).getName(), maxOutput,
        itsKb->getSimpleVisualObject(maxNeuron2).getName(), maxOutput2, meanOutput);

  if (maxOutput > 0.25 &&              // require non-negligible activation
      maxOutput > 1.75 * meanOutput && // require larger than mean
      maxOutput > 1.25 * maxOutput2)   // require larger than second best
    { vo = itsKb->getSimpleVisualObject(maxNeuron); return true; }

  return false;
}

// ######################################################################
bool BPnnet::save(const char* filename) const
{
  char fname[256]; strcpy(fname, filename); strcat(fname, "_w1.raw");
  std::ofstream s(fname, std::ofstream::binary);
  if (s.is_open() == false) { LERROR("Cannot write %s", fname); return false; }
  s.write((char *)(weightFromInput.getArrayPtr()),
          weightFromInput.getSize() * sizeof(double));
  s.close();

  strcpy(fname, filename); strcat(fname, "_w2.raw");
  s.open(fname, std::ofstream::binary);
  if (s.is_open() == false) { LERROR("Cannot write %s", fname); return false; }
  s.write((char *)(weightToOutput.getArrayPtr()),
          weightToOutput.getSize() * sizeof(double));
  s.close();

  return true;
}

// ######################################################################
bool BPnnet::load(const char* filename)
{
  char fname[256]; strcpy(fname, filename); strcat(fname, "_w1.raw");
  std::ifstream s(fname, std::ifstream::binary);
  if (s.is_open() == false) { LERROR("Cannot read %s", fname); return false; }
  s.read((char *)(weightFromInput.getArrayPtr()),
         weightFromInput.getSize() * sizeof(double));
  s.close();

  strcpy(fname, filename); strcat(fname, "_w2.raw");
  s.open(fname, std::ifstream::binary);
  if (s.is_open() == false) { LERROR("Cannot read %s", fname); return false; }
  s.read((char *)(weightToOutput.getArrayPtr()),
         weightToOutput.getSize() * sizeof(double));
  s.close();

  return true;
}

// ######################################################################
void BPnnet::forwardProp( const Image<float> &in)
{
  ASSERT(in.getSize() == numInputUnits);

  // Assign inputs
  for (int i = 0; i < numInputUnits; i ++)
    inputLayer[i].assignInput(in.getVal(i));

  // Forward prop to hidden layer
  double weightedInputSum_h = 0.0;
  for (int n = 0; n < numHiddenUnits; n ++)
    {
      // calculate weighted input sum
      for (int m = 0; m < numInputUnits; m ++)
        {
          double iOutput = inputLayer[m].getActivationLevel();
          double weightedInput = iOutput * weightFromInput.getVal(m, n);
          weightedInputSum_h += weightedInput;
        }
      hiddenLayer[n].assignInput(weightedInputSum_h);
    }

  // Forward prop to output layer
  double weightedInputSum_o = 0.0;
  for (int n = 0; n < numOutputUnits; n ++)
    {
      // calculate weighted input sum
      for (int m = 0; m < numHiddenUnits; m ++)
        {
          double hOutput = hiddenLayer[m].getActivationLevel();
          double weightedInput = hOutput * weightToOutput.getVal(m, n);
          weightedInputSum_o += weightedInput;
        }
      outputLayer[n].assignInput(weightedInputSum_o);
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
