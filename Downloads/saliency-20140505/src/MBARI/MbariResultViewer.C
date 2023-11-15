/*!@file MBARI/MbariResultViewer.C class that manages the results viewing and
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/MbariResultViewer.C $
// $Id: MbariResultViewer.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "MBARI/MbariResultViewer.H"

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "GUI/XWinManaged.H"
#include "Image/CutPaste.H"  // for crop()
#include "Image/Image.H"
#include "Image/ShapeOps.H"  // for rescale()
#include "Image/colorDefs.H"
#include "MBARI/MbariFrameSeries.H"
#include "MBARI/VisualEvent.H"
#include "Media/FrameSeries.H"
#include "Util/Assert.H"
#include "Util/log.H"

#include <cstdio>
#include <fstream>

// Format here is:
//
// { MODOPT_TYPE, "name", &MOC_CATEG, OPTEXP_CORE,
//   "description of what option does",
//   "long option name", 'short option name', "valid values", "default value" }
//

// alternatively, for MODOPT_ALIAS option types, format is:
//
// { MODOPT_ALIAS, "", &MOC_ALIAS, OPTEXP_CORE,
//   "description of what alias does",
//   "long option name", 'short option name', "", "list of options" }
//

// NOTE: do not change the default value of any existing option unless
// you really know what you are doing!  Many components will determine
// their default behavior from that default value, so you may break
// lots of executables if you change it.

// #################### MbariResultViewer options:
// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsaveEvents =
  { MODOPT_ARG_STRING, "MRVsaveEvents", &MOC_MBARIRV, OPTEXP_MRV,
    "Save the event structure to a text file",
    "mbari-save-events", '\0', "fileName", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVloadEvents =
  { MODOPT_ARG_STRING, "MRVloadEvents", &MOC_MBARIRV, OPTEXP_MRV,
    "Load the event structure from a text file "
    "instead of computing it from the frames",
    "mbari-load-events", '\0', "fileName", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsaveProperties =
  { MODOPT_ARG_STRING, "MRVsaveProperties", &MOC_MBARIRV, OPTEXP_MRV,
    "Save the event property vector to a text file",
    "mbari-save-properties", '\0', "fileName", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVloadProperties =
  { MODOPT_ARG_STRING, "MRVloadProperties", &MOC_MBARIRV, OPTEXP_MRV,
    "Load the event property vector from a text file",
    "mbari-load-properties", '\0', "fileName", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsavePositions =
  { MODOPT_ARG_STRING, "MRVsavePositions", &MOC_MBARIRV, OPTEXP_MRV,
    "Save the positions of events to a text file",
    "mbari-save-positions", '\0', "fileName", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVmarkInteresting =
  { MODOPT_ARG(BitObjectDrawMode), "MRVmarkInteresting", &MOC_MBARIRV, OPTEXP_MRV,
    "Way to mark interesting events in output frames of MBARI programs",
    "mbari-mark-interesting", '\0', "<None|Shape|Outline|BoundingBox>",
    "BoundingBox" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVopacity =
  { MODOPT_ARG(float), "MRVopacity", &MOC_MBARIRV, OPTEXP_MRV,
    "Opacity of shape or outline markings of events",
    "mbari-opacity", '\0', "<0.0 ... 1.0>", "1.0" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVmarkCandidate =
  { MODOPT_FLAG, "MRVmarkCandidate", &MOC_MBARIRV, OPTEXP_MRV,
    "Mark candidates for interesting events in output frames of MBARI programs",
    "mbari-mark-candidate", '\0', "", "true" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVmarkPrediction =
  { MODOPT_FLAG, "MRVmarkPrediction", &MOC_MBARIRV, OPTEXP_MRV,
    "Mark the Kalman Filter's prediction for the location of an object "
    "in output frames of MBARI programs",
    "mbari-mark-prediction", '\0', "", "false" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVmarkFOE =
  { MODOPT_FLAG, "MRVmarkFOE", &MOC_MBARIRV, OPTEXP_MRV,
    "Mark the focus of expansion in the output frames of MBARI programs",
    "mbari-mark-foe", '\0', "", "false" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsaveResults =
  { MODOPT_FLAG, "MRVsaveResults", &MOC_MBARIRV, OPTEXP_MRV,
    "Save intermediate results in MBARI programs to disc",
    "mbari-save-results", '\0', "", "false" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVdisplayResults =
  { MODOPT_FLAG, "MRVdisplayResults", &MOC_MBARIRV, OPTEXP_MRV,
    "Display intermediate results in MBARI programs",
    "mbari-display-results", '\0', "", "false" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsaveOutput =
  { MODOPT_FLAG, "MRVsaveOutput", &MOC_MBARIRV, OPTEXP_MRV,
    "Save output frames in MBARI programs",
    "mbari-save-output", '\0', "", "true" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVdisplayOutput =
  { MODOPT_FLAG, "MRVdisplayOutput", &MOC_MBARIRV, OPTEXP_MRV,
    "Display output frames in MBARI programs",
    "mbari-display-output", '\0', "", "false" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVshowEventLabels =
  { MODOPT_FLAG, "MRVshowEventLabels", &MOC_MBARIRV, OPTEXP_MRV,
    "Write event labels into the output frames",
    "mbari-label-events", '\0', "", "true" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVrescaleDisplay =
  { MODOPT_ARG(Dims), "MRVrescaleDisplay", &MOC_MBARIRV, OPTEXP_MRV,
    "Rescale displays to <width>x<height>, or 0x0 for no rescaling",
    "mbari-rescale-display", '\0', "<width>x<height>", "0x0" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsaveEventNums =
  { MODOPT_ARG_STRING, "MRVsaveEventNums", &MOC_MBARIRV, OPTEXP_MRV,
    "Save video clips showing specific events",
    "mbari-save-event-clip", '\0', "ev1,ev1,...,evN; or: all", "" };

// Used by: MbariResultViewer
const ModelOptionDef OPT_MRVsizeAvgCache =
  { MODOPT_ARG(int), "MRVsizeAvgCache", &MOC_MBARIRV, OPTEXP_MRV,
    "The number of frames used to compute the running average",
    "mbari-cache-size", '\0', "<int>", "10" };


// #############################################################################
MbariResultViewer::MbariResultViewer(ModelManager& mgr,
                                     nub::soft_ref<OutputMbariFrameSeries> ofs)
  : ModelComponent(mgr, std::string("MbariResultViewer"),
                   std::string("MbariResultViewer")),
    itsSaveResults(&OPT_MRVsaveResults, this),
    itsDisplayResults(&OPT_MRVdisplayResults, this),
    itsMarkInteresting(&OPT_MRVmarkInteresting, this),
    itsOpacity(&OPT_MRVopacity, this),
    itsMarkCandidate(&OPT_MRVmarkCandidate, this),
    itsMarkPrediction(&OPT_MRVmarkPrediction, this),
    itsMarkFOE(&OPT_MRVmarkFOE, this),
    itsSaveOutput(&OPT_MRVsaveOutput, this),
    itsDisplayOutput(&OPT_MRVdisplayOutput, this),
    itsShowEventLabels(&OPT_MRVshowEventLabels, this),
    itsRescaleDisplay(&OPT_MRVrescaleDisplay, this),
    itsSizeAvgCache(&OPT_MRVsizeAvgCache, this),
    itsSaveEventsName(&OPT_MRVsaveEvents, this),
    itsLoadEventsName(&OPT_MRVloadEvents, this),
    itsSavePropertiesName(&OPT_MRVsaveProperties, this),
    itsLoadPropertiesName(&OPT_MRVloadProperties, this),
    itsSavePositionsName(&OPT_MRVsavePositions, this),
    itsSaveEventNumString(&OPT_MRVsaveEventNums, this),
    resFrameWindow(NULL),
    itsOfs(ofs),
    colInteresting(COL_INTERESTING),
    colCandidate(COL_CANDIDATE),
    colPrediction(COL_PREDICTION),
    colFOE(COL_FOE)
{
  // need to register the OutputMbariFrameSeries with the ModelManager?
  if (!mgr.hasSubComponent(ofs)) mgr.addSubComponent(ofs);
}

// #############################################################################
MbariResultViewer::~MbariResultViewer()
{
  // destroy everything
  freeMem();
}

// ######################################################################
void MbariResultViewer::paramChanged(ModelParamBase* const param,
                                     const bool valueChanged,
                                     ParamClient::ChangeStatus* status)
{
  ModelComponent::paramChanged(param, valueChanged, status);

  // if the param is out itsMarkInteresting set the color accordingly
  //if (param == &itsMarkInteresting)
  //{
  //  if (itsMarkInteresting.getVal()) colInteresting = COL_INTERESTING;
  //  else colInteresting = COL_TRANSPARENT;
  //}

  // if the param is out itsMarkCandidate set the color accordingly
  if (param == &itsMarkCandidate)
    {
      if (itsMarkCandidate.getVal()) colCandidate = COL_CANDIDATE;
      else colCandidate = COL_TRANSPARENT;
    }

  // if the param is out itsMarkSkip set the color accordingly
  else if (param == &itsMarkPrediction)
    {
      if (itsMarkPrediction.getVal()) colPrediction = COL_PREDICTION;
      else colPrediction = COL_TRANSPARENT;
    }

  // if the param is out itsMarkSkip set the color accordingly
  else if (param == &itsMarkFOE)
    {
      if (itsMarkFOE.getVal()) colFOE = COL_FOE;
      else colFOE = COL_TRANSPARENT;
    }

  // if the param is itsSaveEventNum, parse the string and fill the vector
  else if (param == &itsSaveEventNumString)
    parseSaveEventNums(itsSaveEventNumString.getVal());
}

// ######################################################################
void MbariResultViewer::reset1()
{
  // destroy our stuff
  freeMem();

  // propagate to our base class:
  ModelComponent::reset1();
}

// ######################################################################
void MbariResultViewer::freeMem()
{
  for (uint i = 0; i < itsResultWindows.size(); ++i)
    if (itsResultWindows[i] != NULL) delete itsResultWindows[i];

  if (resFrameWindow != NULL) delete resFrameWindow;

  itsResultNames.clear();
  itsResultWindows.clear();

  itsSaveEventsName.setVal("");
  itsLoadEventsName.setVal("");
  itsSavePropertiesName.setVal("");
  itsLoadPropertiesName.setVal("");
}

// #############################################################################
template <class T>
void MbariResultViewer::output(const Image<T>& img, const uint frameNum,
                               const std::string& resultName, const int resNum)
{
  if (itsDisplayResults.getVal()) display(img, frameNum, resultName, resNum);
  if (itsSaveResults.getVal()) save(img, frameNum, resultName, resNum);
}

// #############################################################################
template <class T>
void MbariResultViewer::display(const Image<T>& img, const uint frameNum,
                                const std::string& resultName, const int resNum)
{
  uint num = getNumFromString(resultName);
  itsResultWindows[num] = displayImage(img, itsResultWindows[num],
                                       getLabel(num, frameNum, resNum).c_str());
}

// #############################################################################
void MbariResultViewer::save(const Image< PixRGB<byte> >& img,
                             const uint frameNum,
                             const std::string& resultName,
                             const int resNum)
{
  //uint num = getNumFromString(resultName);
  itsOfs->writeMbariRGB(img,getFileStem(resultName, resNum),frameNum);
}

// #############################################################################
void MbariResultViewer::save(const Image<byte>& img,
                             const uint frameNum,
                             const std::string& resultName,
                             const int resNum)
{
  //uint num = getNumFromString(resultName);
  itsOfs->writeMbariGray(img,getFileStem(resultName, resNum),frameNum);
}

// #############################################################################
void MbariResultViewer::save(const Image<float>& img,
                             const uint frameNum,
                             const std::string& resultName,
                             const int resNum)
{
  //uint num = getNumFromString(resultName);
  itsOfs->writeMbariFloat(img,getFileStem(resultName, resNum),
                          FLOAT_NORM_0_255,frameNum);
}

// #############################################################################
void MbariResultViewer::outputResultFrame(const Image< PixRGB<byte> >& resultImg,
                                          const std::string& frameStem,
                                          const uint frameNum,
                                          VisualEventSet& evts,
                                          PropertyVectorSet& pvs,
                                          const int circleRadius)
{
  Image< PixRGB<byte> > img = resultImg;

  if (itsDisplayOutput.getVal() || itsSaveOutput.getVal())
    evts.drawTokens(img, frameNum, pvs, circleRadius,
                    itsMarkInteresting.getVal(), itsOpacity.getVal(),
                    colInteresting, colCandidate, colPrediction,
                    colFOE, itsShowEventLabels.getVal());

  // display the frame?
  if (itsDisplayOutput.getVal())
    {
      // make the label
      char label[2048];
      sprintf(label,"%s%06d",itsOfs->getFileStem().c_str(),frameNum);

      resFrameWindow = displayImage(img,resFrameWindow,label);
    }

  // save the resulting frame to disk ?
  if (itsSaveOutput.getVal()) itsOfs->writeMbariRGB(img,frameStem,frameNum);
}

// #############################################################################
bool MbariResultViewer::isLoadEventsNameSet() const
{
  return (itsLoadEventsName.getVal().length() > 0);
}


// #############################################################################
void MbariResultViewer::loadVisualEventSet(VisualEventSet& ves) const
{
  std::ifstream ifs(itsLoadEventsName.getVal().c_str());
  ves.readFromStream(ifs);
  ifs.close();
}
// #############################################################################
bool MbariResultViewer::isLoadPropertiesNameSet() const
{
  return (itsLoadPropertiesName.getVal().length() > 0);
}

// #############################################################################
void MbariResultViewer::loadProperties(PropertyVectorSet& pvs) const
{
  std::ifstream ifs(itsLoadPropertiesName.getVal().c_str());
  pvs.readFromStream(ifs);
  ifs.close();
}

// #############################################################################
bool MbariResultViewer::isSaveEventsNameSet() const
{
  return (itsSaveEventsName.getVal().length() > 0);
}

// #############################################################################
void MbariResultViewer::saveVisualEventSet(const VisualEventSet& ves) const
{
  std::ofstream ofs(itsSaveEventsName.getVal().c_str());
  ves.writeToStream(ofs);
  ofs.close();
}

// #############################################################################
bool MbariResultViewer::isSavePropertiesNameSet() const
{
  return (itsSavePropertiesName.getVal().length() > 0);
}

// #############################################################################
void MbariResultViewer::saveProperties(const PropertyVectorSet& pvs) const
{
  std::ofstream ofs(itsSavePropertiesName.getVal().c_str());
  pvs.writeToStream(ofs);
  ofs.close();
}

// #############################################################################
bool MbariResultViewer::isSavePositionsNameSet() const
{
  return (itsSavePositionsName.getVal().length() > 0);
}

// #############################################################################
void MbariResultViewer::savePositions(const VisualEventSet& ves) const
{
  std::ofstream ofs(itsSavePositionsName.getVal().c_str());
  ves.writePositions(ofs);
  ofs.close();
}

// #############################################################################
bool MbariResultViewer::needFrames() const
{
  bool needOutput = itsSaveOutput.getVal() || itsDisplayOutput.getVal();
  bool needInput = !isLoadEventsNameSet() || isSaveEventClip();
  return (needInput || needOutput);
}

// #############################################################################
uint MbariResultViewer::getAvgCacheSize() const
{
  return itsSizeAvgCache.getVal();
}

// #############################################################################
bool MbariResultViewer::isSaveEventClip() const
{
  return (itsSaveEventNums.size() > 0);
}

// #############################################################################
uint MbariResultViewer::numSaveEventClips() const
{
  return itsSaveEventNums.size();
}

// #############################################################################
uint MbariResultViewer::getSaveEventClipNum(uint idx) const
{
  ASSERT(idx < itsSaveEventNums.size());
  return itsSaveEventNums[idx];
}

// #############################################################################
uint MbariResultViewer::getNumFromString(const std::string& resultName)
{
  // see if we can find this guy in our list
  for (uint i = 0; i < itsResultNames.size(); ++i)
    if (itsResultNames[i].compare(resultName) == 0)
      return i;

  // didn't find it -> make a new entry and return index of this new entry
  itsResultNames.push_back(resultName);
  itsResultWindows.push_back(NULL);

  return (itsResultNames.size() - 1);
}

// #############################################################################
std::string MbariResultViewer::getLabel(const uint num, const uint frameNum,
                                        const int resNum)
{
  ASSERT(num < itsResultNames.size());
  char fnum[7];
  sprintf(fnum,"%06d",frameNum);
  return (getFileStem(itsResultNames[num], resNum) + std::string(fnum));
}

// #############################################################################
std::string MbariResultViewer::getFileStem(const std::string& resultName,
                                           const int resNum)
{
  char rnum[4];
  if (resNum >= 0) sprintf(rnum,"%02d_",resNum);
  else sprintf(rnum,"_");

  return(itsOfs->getFileStem() + std::string("_")
         + resultName + std::string(rnum));
}

// #############################################################################
template <class T>
XWinManaged* MbariResultViewer::displayImage(const Image<T>& img,
                                             XWinManaged* win,
                                             const char* label)
{
  // need to rescale?
  Dims dims = itsRescaleDisplay.getVal();
  if (dims.isEmpty()) dims = img.getDims();
  bool doRescale = (dims != img.getDims());

  // does the window have to be re-constructed?
  if (win != NULL)
    {
      if (win->getDims() != dims) delete win;
    }

  if (win == NULL)
    {
      if (doRescale)
        win = new XWinManaged(rescale(img, dims),label);
      else
        win = new XWinManaged(img,label);
    }
  else
    {
      if (doRescale)
        win->drawImage(rescale(img,dims));
      else
        win->drawImage(img);

      win->setTitle(label);
    }

  return win;
}

// #############################################################################
void MbariResultViewer::saveSingleEventFrame(const Image< PixRGB<byte> >& img,
                                             int frameNum,
                                             const VisualEvent& event)
{
  ASSERT(event.isFrameOk(frameNum));

  // create the file stem
  char evnum[10];
  sprintf(evnum,"_evt%04d_",event.getEventNum());
  std::string filestem = itsOfs->getFileStem() + std::string(evnum);

  const int pad = 10;
  Dims maxDims = event.getMaxObjectDims();
  Dims d(maxDims.w() + 2 * pad, maxDims.h() + 2 * pad);

  // compute the correct bounding box and cut it out
  Rectangle bbox = event.getToken(frameNum).bitObject.getBoundingBox();
  //Point2D<int> cen = event.getToken(frameNum).bitObject.getCentroid();

  // first the horizontal direction
  int wpad = (d.w() - bbox.width()) / 2;
  int ll = bbox.left() - wpad;
  //int ll = cen.i - d.w() / 2;
  int rr = ll + d.w();
  if (ll < 0) { rr -= ll; ll = 0; }
  if (rr >= img.getWidth()) { rr = img.getWidth() - 1; ll = rr - d.w(); }

  // now the same thing with the vertical direction
  int hpad = (d.h() - bbox.height()) / 2;
  int tt = bbox.top() - hpad;
  //int tt = cen.j - d.h() / 2;
  int bb = tt + d.h();
  if (tt < 0) { bb -= tt; tt = 0; }
  if (bb >= img.getHeight()) { bb = img.getHeight() - 1; tt = bb - d.h(); }

  // cut out the rectangle and save it
  Image< PixRGB<byte> > cut = crop(img, Rectangle::tlbrI(tt,ll,bb,rr));
  itsOfs->writeMbariRGB(cut, filestem, frameNum);
}

// #############################################################################
void MbariResultViewer::parseSaveEventNums(const std::string& value)
{
  itsSaveEventNums.clear();

  // format here is "c,...,c"
  int curpos = 0, len = value.length();
  while (curpos < len)
    {
      // get end of next number
      int nextpos = value.find_first_not_of("-.0123456789eE",curpos);
      if (nextpos == -1) nextpos = len;

      // no number characters found -> bummer
      if (nextpos == curpos)
        LFATAL("Error parsing the SaveEventNum string '%s' - found '%c' "
               "instead of a number.",value.c_str(),value[curpos]);

      // now let's see - can we get a number here?
      uint evNum;
      int rep = sscanf(value.substr(curpos,nextpos-curpos).c_str(),"%i",&evNum);

      // couldn't read a number -> bummer
      if (rep != 1)
        LFATAL("Error parsing SaveEventNum string '%s' - found '%s' instead of "
               "a number.", value.c_str(),
               value.substr(curpos,nextpos-curpos).c_str());

      // yeah! found a number -> store it
      itsSaveEventNums.push_back(evNum);

      LDEBUG("evNum = %i; value[nextpos] = '%c'",evNum,value[nextpos]);

      // not a comma -> bummer
      if ((nextpos < len) && (value[nextpos] != ','))
        LFATAL("Error parsing the SaveEventNum string '%s' - found '%c' "
               "instead of ','.",value.c_str(),value[nextpos]);

      // the character right after the comma should be a number again
      curpos = nextpos + 1;
    }

  // end of string, done
  return;
}
// #############################################################################
// Instantiations
#define INSTANTIATE(T) \
template void MbariResultViewer::output(const Image< T >& img, \
                                        const uint frameNum, \
                                        const std::string& resultName, \
                                        const int resNum); \
template void MbariResultViewer::display(const Image< T >& img, \
                                         const uint frameNum, \
                                         const std::string& resultName, \
                                         const int resNum); \
template XWinManaged* MbariResultViewer::displayImage(const Image< T >& img, \
                                                      XWinManaged* win, \
                                                      const char* label);

INSTANTIATE(PixRGB<byte>);
INSTANTIATE(byte);
INSTANTIATE(float);


// #############################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
