/*!@file SceneUnderstanding/SceneUnderstanding.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/SceneUnderstanding.H $
// $Id: SceneUnderstanding.H 13413 2010-05-15 21:00:11Z itti $
//

#ifndef SCENEUNDERSTANDING_SCENEUNDERSTANDING_H_DEFINED
#define SCENEUNDERSTANDING_SCENEUNDERSTANDING_H_DEFINED

#include "Util/Types.H" // for uint
#include "Image/Point2D.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Component/ModelManager.H"
#include "Learn/Bayes.H"
#include "Learn/SWIProlog.H"
#include "Channels/DescriptorVec.H"
#include "Channels/SingleChannel.H"
#include "Channels/ComplexChannel.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/SimulationViewer.H"
#include "plugins/SceneUnderstanding/WorkingMemory.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include <vector>
#include <string>

class SceneUnderstanding
{
public:

  SceneUnderstanding(ModelManager *mgr, nub::ref<StdBrain> &brain);

  //! Destructor
  ~SceneUnderstanding();

  //! get the brain
  nub::ref<StdBrain> getBrainPtr();

  //! Get the Bayes network
  Bayes* getBayesPtr();

  //! get the Prolog engine
  SWIProlog* getPrologPtr();

  //! get the descriptor vector
  DescriptorVec* getDescriptorVecPtr();

  //! Set the current image
  void setImage(Image<PixRGB<byte> > &img);

  //! evolve the brain
  //! Return the intrest level
  float evolveBrain();

  //! Get the current fovea location (attention location)
  Point2D<int> getFoveaLoc();

  //! Get a feature vector from the fovea at location p
  std::vector<double> getFV(Point2D<int> foveaLoc);

  //! classify the object under the fovea positioned at p
  int classifyFovea(Point2D<int> foveaLoc, double *prob=NULL);

  //! Learn to associate the feature under fovea with object
  void learn(Point2D<int> foveaLoc, const char *name);

  //! Try to bias the feature maps to locate object
  //! Return true if we know about the object
  bool biasFor(const char *name);

  //! Recognize using high order information. in particuler use prolog KB
  std::string highOrderRec();

private:

  ModelManager *itsMgr;
  nub::ref<StdBrain> itsBrain;
  nub::soft_ref<SimEventQueue> itsSEQ; //gets set from the manager
  Bayes *itsBayes;
  SWIProlog *itsProlog;
  DescriptorVec* itsDescriptorVec;
  Image<PixRGB<byte> > itsImg;      //Out current image on the retina
  Point2D<int> itsCurrentFoveaLoc;     //The currentFovea Location
  WorkingMemory *itsWorkingMemory; //Our working memory


};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
