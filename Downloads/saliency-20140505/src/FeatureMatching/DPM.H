/*!@file Features/DPM.H  deformable part models */
//Code derived from http://www.cs.uchicago.edu/~pff/latent/
//author = "Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D.",
//title = "Discriminatively Trained Deformable Part Models, Release 4",
//howpublished = "http://people.cs.uchicago.edu/~pff/latent-release4/"}

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL$
// $Id$
//

#ifndef DPM_H_DEFINED
#define DPM_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Features/HOG.H"
#include "Util/JobServer.H"
#include "Util/JobWithSemaphore.H"
#include "Util/WorkThreadServer.H"

#include <vector>
#include <string>
#include <stdio.h>

class DPM
{
public:


  struct HOGFeatures
  {
    ImageSet<double> features;
    int bins;
    double scale;
  };

  struct ModelScore 
  {
    int level; //Level int the pyramid
    Image<double> score; //the max Score between components
    Image<int> component; //the comp that produced this score

    ModelScore(const Image<double> s, const Image<int> comp, int l) :
      level(l), score(s), component(comp)
    {
    }
  };

  struct Detection
  {
    Rectangle bb; //the bounding box
    double score; //the score
    int component; //the component that lead to this score

    Detection(const Rectangle rect, double s, int c) :
      bb(rect),
      score(s),
      component(c)
    {
    }
    Detection() : 
      score(0),
      component(-1)
    {}
  };


  struct ModelPart
  {
    ImageSet<double> features;
    Point2D<float> anchor;
    float scale;  //The scale from which the anchor is set
    std::vector<double> deformation;
  };

  struct ModelComponent
  {
    ImageSet<double> rootFilter;
    double offset;
    std::vector<ModelPart> parts;
  };

  struct Model
  {
    std::vector<ModelComponent> components;
  };
  
  //! Constructor
  DPM();

  //! Destructor
  virtual ~DPM();

  void computeFeaturePyramid(const Image<PixRGB<byte> >& img);
  
  //! Load a model from a file
  void readModel(const char* fileName);

  //! Get an image showing the model
  Image<PixRGB<byte> > getModelImage();

  //! Convolve the whole model
  void convolveModel();

  //! Convolve just a component at a particular level in the pyramid
  Image<double> convolveComponent(const int comp, const int level);
  
  
  //! Get bounding boxes
  std::vector<Detection> getBoundingBoxes(const float thresh);

  //! Preform filtering using non maximum suppression
  std::vector<Detection> filterDetections(const std::vector<Detection>& detections, const float overlap);
  
  Image<double> convolveFeatures(const ImageSet<double>& imgFeatures, 
      const ImageSet<double>& filterFeatures);
 
  //! Distance transform for deformation
  Image<double> distanceTrans(const Image<double>& score,
      const std::vector<double>& deformation);
 
  //! Distance transform helper
  void dtHelper(const Image<double>::const_iterator src,
      Image<double>::iterator dst,
      Image<int>::iterator ptr,
      int step,
      int s1, int s2, int d1, int d2,
      double a, double b);
  
  
protected:
  
  class DPMJob : public JobWithSemaphore
  {
    public:
      DPMJob(DPM* dpm, const int comp, const int l) :
        itsDPM(dpm), itsComponent(comp), itsLevel(l) 
    { }

      virtual ~DPMJob() { }

      virtual void run()
      {
        itsScore = itsDPM->convolveComponent(itsComponent, itsLevel);
        this->markFinished();
      }

      Image<double> getScore() { return itsScore; }
      int getComponent() { return itsComponent; }

      virtual const char* jobType() const { return "DPMJob"; }

    private:
      DPM* itsDPM;
      int itsComponent;
      int itsLevel;
      Image<double> itsScore;
  };


private:
  std::vector<HOGFeatures> itsFeaturesPyramid;
  Model itsModel;

  int itsInterval; //the interval between features in the pyramid

  std::vector<ModelScore> itsModelScores; //Scores of the detections

  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
  
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
