/*!@file SceneUnderstanding/LGN.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/LGN.H $
// $Id: LGN.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef LGN_H_DEFINED
#define LGN_H_DEFINED

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
#include "Raster/GenericFrame.H"

#include <vector>
#include <string>


class LGN : public SimModule
{
public:

  enum CHAN {LUM, RG, BY};

  //! Constructor
  /*! See ModelComponent.H for details */
  LGN(OptionManager& mgr, const std::string& descrName = "LGN",
      const std::string& tagName = "LGN");

  //! Destructor
  virtual ~LGN();

  void init(Dims numCells);
  void evolve();

  ImageSet<float> getInput() {return itsCellsInput;}
  ImageSet<float> getLGNCells() {return itsCellsMu;}
  void setBias(const ImageSet<float> &biasImg);


  //! Get a the input,stats,and output
  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new input frame is available
  SIMCALLBACK_DECLARE(LGN, SimEventInputFrame);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(LGN, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  bool itsInitialized; //are we initalized
  Image<PixRGB<byte> > itsCurrentImg;
  ImageSet<float> itsCellsMu; //Current perception of the stimulus
  ImageSet<float> itsCellsSig; //Current perception of the stimulus
  ImageSet<float> itsCellsInput; //The input to the cells
};


/* ############################### LGN sim events ######################## */
class SimEventLGNOutput : public SimEvent
{
public:
  SimEventLGNOutput(SimModule* src, ImageSet<float>& cellsOutput,
      rutz::shared_ptr<GenericFrame::MetaData>& metaData) :
    SimEvent(src), itsCells(cellsOutput), itsMetaData(metaData)
  {}

  virtual ~SimEventLGNOutput(){}

  const ImageSet<float> getCells() { return itsCells; }
  rutz::shared_ptr<GenericFrame::MetaData> getMetaData() { return itsMetaData; }


private:
  const ImageSet<float>& itsCells;
  rutz::shared_ptr<GenericFrame::MetaData> itsMetaData;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
