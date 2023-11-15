/*!@file SceneUnderstanding/TensorVoting.H  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/TensorVoting.H $
// $Id: TensorVoting.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef TensorVoting_H_DEFINED
#define TensorVoting_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
//#include "Image/OpenCVUtil.H"
#include "Image/TensorOps.H"

#include <vector>
#include <string>

//Some functions inspired from a matlab version of tensor voting by Trevor Linton
// (http://www.mathworks.com/matlabcentral/fileexchange/21051-tensor-voting-framework)

class TensorVoting
{
public:

  TensorVoting();

  //! Destructor
  ~TensorVoting();

  void evolve();
  TensorField evolve(const Image<PixRGB<byte> >& img);
  TensorField evolve(const TensorField& tf, bool performNonMaxSurp=false);

  TensorField getTensorField() { return itsTensorField; }
  Image<float> getTokensMag(bool normalize);

  inline const Dims& getDims() const { return itsTensorField.t1.getDims(); }

protected:

  TensorField transImage(Image<float>& img);
  TensorField findFeatures(TensorField& tensorField, float sigma);

  TensorField createBallTensorField(float sigma);
  TensorField createStickTensorField(float u, float v, float sigma);
  void getBallVotes(Image<float>& im,
                           TensorField& tensorField, float sigma);
  TensorField getStickVotes(const TensorField& tensorField, float sigma);
  TensorField getStickVotes2(const TensorField& tensorField, float sigma);


  TensorField calcSparseField(Image<float>& img);
  TensorField calcRefinedField(TensorField& tensorField,
      Image<float>& img, float sigma);

  //Create the tensor fields first, so we can save time latter
  void createTensorFields(float sigma);

  void generateImage();



private:
  float itsSigma;
  Image<float> inputImg;
  std::vector<TensorField> itsStickFields;
  TensorField itsBallField;
  TensorField itsTensorField;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
