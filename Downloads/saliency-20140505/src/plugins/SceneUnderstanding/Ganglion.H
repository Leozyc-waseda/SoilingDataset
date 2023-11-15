/*!@file SceneUnderstanding/Ganglion.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Ganglion.H $
// $Id: Ganglion.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef GANGLION_H_DEFINED
#define GANGLION_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Channels/InputFrame.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"

#include <vector>
#include <string>


class Ganglion : public SimModule
{
public:

  enum CHAN {LUM, RG, BY};

  //! Constructor
  /*! See ModelComponent.H for details */
  Ganglion(OptionManager& mgr, const std::string& descrName = "Ganglion",
      const std::string& tagName = "Ganglion");

  //! Destructor
  virtual ~Ganglion();

  void init(Dims numCells);
  void setInput(const Image<PixRGB<byte> > &img);
  void evolve();

  ImageSet<float> getOutput() { return itsGanglionCellsOutput; }
  ImageSet<float> getInput() {return itsGanglionCellsInput;}
  ImageSet<float> getGanglionCells() {return itsGanglionCellsMu;}
  void setBias(const ImageSet<float> &biasImg);


  //! Get a the input,stats,and output
  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new input frame is available
  SIMCALLBACK_DECLARE(Ganglion, SimEventInputFrame);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Ganglion, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  Image<PixRGB<byte> > itsCurrentImg;
  ImageSet<float> itsGanglionCellsPrior; //Current perception of the stimulus
  ImageSet<float> itsGanglionCellsMu; //Current perception of the stimulus
  ImageSet<float> itsGanglionCellsSig; //Current perception of the stimulus
  ImageSet<float> itsGanglionCellsInput; //The input to the cells
  ImageSet<float> itsGanglionCellsOutput; //The output of the cells
  Point2D<int> itsProbe;
};


/* ############################### Ganglion sim events ######################## */
class SimEventGanglionOutput : public SimEvent
{
public:
  SimEventGanglionOutput(SimModule* src, ImageSet<float>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventGanglionOutput(){}

  const ImageSet<float> getCells() { return itsCells; }

private:
  const ImageSet<float>& itsCells;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
