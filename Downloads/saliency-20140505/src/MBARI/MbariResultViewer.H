/*!@file MBARI/MbariResultViewer.H class that manages the results viewing and
  saving for the MBARI programs */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/MbariResultViewer.H $
// $Id: MbariResultViewer.H 8160 2007-03-21 21:34:16Z rjpeters $
//

#ifndef MBARIRESULTVIEWER_H_DEFINED
#define MBARIRESULTVIEWER_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/ArrayData.H"
#include "Image/Pixels.H"
#include "MBARI/BitObjectDrawModes.H"
#include "Util/Types.H"
#include "rutz/shared_ptr.h"
#include <string>
#include <vector>

class VisualEvent;
class VisualEventSet;
class PropertyVectorSet;
class OutputMbariFrameSeries;
class ModelManager;
class XWinManaged;
template <class T> class Image;

//! Command-line options for MbariResultViewer
//@{
extern const ModelOptionDef OPT_MRVsaveEvents;
extern const ModelOptionDef OPT_MRVloadEvents;
extern const ModelOptionDef OPT_MRVsaveProperties;
extern const ModelOptionDef OPT_MRVloadProperties;
extern const ModelOptionDef OPT_MRVsavePositions;
extern const ModelOptionDef OPT_MRVmarkInteresting;
extern const ModelOptionDef OPT_MRVopacity;
extern const ModelOptionDef OPT_MRVmarkCandidate;
extern const ModelOptionDef OPT_MRVmarkPrediction;
extern const ModelOptionDef OPT_MRVmarkFOE;
extern const ModelOptionDef OPT_MRVsaveResults;
extern const ModelOptionDef OPT_MRVdisplayResults;
extern const ModelOptionDef OPT_MRVsaveOutput;
extern const ModelOptionDef OPT_MRVdisplayOutput;
extern const ModelOptionDef OPT_MRVshowEventLabels;
extern const ModelOptionDef OPT_MRVrescaleDisplay;
extern const ModelOptionDef OPT_MRVsaveEventNums;
extern const ModelOptionDef OPT_MRVsizeAvgCache;
//@}

//! Class to facilitate displaying and saving of intermediate results
/*! This class provides command lne options to control the saving
  and displaying of intermediate results as well as output frames
  for MBARI programs */
class MbariResultViewer : public ModelComponent
{
public:
  //! Constructor - give a ModelManager and an OutputMbariFrameSeries
  /*! If the OutputMbariFrameSeries is not yet registered in the manager,
    it will be registered here */
  MbariResultViewer(ModelManager& mgr, nub::soft_ref<OutputMbariFrameSeries> ofs);

  //! Destructor - destroy all the windows that may be present
  ~MbariResultViewer();

  //! implements ModelComponent::paramChanged
  /*! intercepts setting of MRVmarkXXX variables and sets drawing
    colors accordingly */
  virtual void paramChanged(ModelParamBase* const param,
                            const bool valueChanged,
                            ParamClient::ChangeStatus* status);

  //! destroy all the windows
  virtual void reset1();

  //! save and/or display intermediate result
  /*! whether the result image is saved and/or displayed is
    determined by the ModelParams
    @param img the image containing the intermediate result
    @param frameNum the frame number of the image
    @param resultName a string that defines the kind of interemediate
    result - this is appended to the filestem for the file name
    and/or window title
    @param resNum if there are several results of this type for each frame,
    the number of the result*/
  template <class T>
  void output(const Image<T>& img, const uint frameNum,
              const std::string& resultName, const int resNum = -1);

  //! display intermediate result
  /*!@param img the image containing the intermediate result
    @param frameNum the frame number of the image
    @param resultName a string that defines the kind of interemediate
    result - this is appended to the filestem for the file name
    and/or window title
    @param resNum if there are several results of this type for each frame,
    the number of the resul*/
  template <class T>
  void display(const Image<T>& img, const uint frameNum,
               const std::string& resultName, const int resNum = -1);

  //! save intermediate result
  /*!@param img the image containing the intermediate result
    @param frameNum the frame number of the image
    @param resultName a string that defines the kind of interemediate
    result - this is appended to the filestem for the file name
    and/or window title
    @param resNum if there are several results of this type for each frame,
    the number of the resul*/
  void save(const Image< PixRGB<byte> >& img, const uint frameNum,
            const std::string& resultName, const int resNum = -1);

  //! save intermediate result
  /*!@param img the image containing the intermediate result
    @param frameNum the frame number of the image
    @param resultName a string that defines the kind of interemediate
    result - this is appended to the filestem for the file name
    and/or window title
    @param resNum if there are several results of this type for each frame,
    the number of the resul*/
  void save(const Image<byte>& img, const uint frameNum,
            const std::string& resultName, const int resNum = -1);

  //! save intermediate result
  /*!@param img the image containing the intermediate result
    @param frameNum the frame number of the image
    @param resultName a string that defines the kind of interemediate
    result - this is appended to the filestem for the file name
    and/or window title
    @param resNum if there are several results of this type for each frame,
    the number of the resul*/
  void save(const Image<float>& img, const uint frameNum,
            const std::string& resultName, const int resNum = -1);

  //! save and possibly display an output frame
  /*!Depending on the ModelParams, interesting events, candidates
    for interesting events and skipped event locations are marked
    in the frame, the frame is displayed (if desired) and saved.
    @param resultImg the output image
    @param frameNum the frame number of the output image
    @param evts the event set to be used for drawing the events
    @param circleRadius the radius of the circles used for marking*/
  void outputResultFrame(const Image< PixRGB<byte> >& resultImg,
                         const std::string& frameStem,
                         const uint frameNum,
                         VisualEventSet& evts,
                         PropertyVectorSet& pvs,
                         const int circleRadius);

  //! check if the LoadEventsName is set
  bool isLoadEventsNameSet() const;

  //! load the VisualEventSet from the file LoadEventsName
  void loadVisualEventSet(VisualEventSet& ves) const;

  //! check if the LoadPropertiesName is set
  bool isLoadPropertiesNameSet() const;

  //! load the PropertyVector from the file LoadPropertiesName
  void loadProperties(PropertyVectorSet& pvs) const;

  //! check if the SaveEventsName is set
  bool isSaveEventsNameSet() const;

  //! save the VisualEventSet to the file SaveEventsName
  void saveVisualEventSet(const VisualEventSet& ves) const;

  //! check if the SavePropertiesName is set
  bool isSavePropertiesNameSet() const;

  //! save the PropertyVector to the file SavePropertiesName
  void saveProperties(const PropertyVectorSet& pvs) const;

  //! check if the SavePositionsName is set
  bool isSavePositionsNameSet() const;

  //! save the positions to the file SavePositionsName
  void savePositions(const VisualEventSet& ves) const;

  //! determine whether the frames actually need to be loaded
  bool needFrames() const;

  //! returns the size of the cache for running averages (from the command line)
  uint getAvgCacheSize() const;

  //! determine whether we have to save any event clips
  bool isSaveEventClip() const;

  //! return the number of event clips we have to save
  uint numSaveEventClips() const;

  //! returns the event number with idx to be saved as a clip
  uint getSaveEventClipNum(uint idx) const;

  //! save a cut-out piece of frame containing a single event
  void saveSingleEventFrame(const Image< PixRGB<byte> >& img,
                            int frameNum, const VisualEvent& event);

protected:

  //! destroy windows and other internal variables
  void freeMem();

  //! get an internal reference number from a resultName
  /*! If this resultName is new, a new reference is created*/
  uint getNumFromString(const std::string& resultName);

  //! make a label text from the reference number and the frame number
  std::string getLabel(const uint num, const uint frameNum, const int resNum = -1);

  //! make the file stem for saving files
  std::string getFileStem(const std::string& resultName, const int resNum = -1);

  //! draw an image to a window, deal with creating and rescaling the window
  template <class T>
  XWinManaged* displayImage(const Image<T>& img,
                            XWinManaged* win,
                            const char* label);

  //! parse the SaveEventNumString and store the numbers in itsSaveEventNums
  void parseSaveEventNums(const std::string& value);

  OModelParam<bool> itsSaveResults;     //!<whether intermediate results are saved
  OModelParam<bool> itsDisplayResults;  //!<whether intermediate results are displayed
  OModelParam<BitObjectDrawMode> itsMarkInteresting; //!<the way interesting events are marked
  OModelParam<float> itsOpacity;        //!<opacity of shape or outline markings of events
  OModelParam<bool> itsMarkCandidate;   //!<whether candidates for int. events are marked
  OModelParam<bool> itsMarkPrediction;  //!<whether predictions of event locations are marked
  OModelParam<bool> itsMarkFOE;         //!<whether the focus of expansion is marked
  OModelParam<bool> itsSaveOutput;      //!<whether the output frames are saved
  OModelParam<bool> itsDisplayOutput;   //!<whether the output frames are displayed
  OModelParam<bool> itsShowEventLabels; //!<whether the event labels are written into output frames

  OModelParam<Dims> itsRescaleDisplay;  //!<rescale intermediate results to this scale for display

  OModelParam<uint> itsSizeAvgCache;     //!<the number of images used for running average

  OModelParam<std::string> itsSaveEventsName;
  OModelParam<std::string> itsLoadEventsName;
  OModelParam<std::string> itsSavePropertiesName;
  OModelParam<std::string> itsLoadPropertiesName;
  OModelParam<std::string> itsSavePositionsName;

  OModelParam<std::string> itsSaveEventNumString;

private:
  std::vector<std::string> itsResultNames;
  std::vector<XWinManaged*> itsResultWindows;
  XWinManaged* resFrameWindow;
  nub::soft_ref<OutputMbariFrameSeries> itsOfs;
  PixRGB<byte> colInteresting, colCandidate, colPrediction, colFOE;
  std::vector<uint> itsSaveEventNums;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
