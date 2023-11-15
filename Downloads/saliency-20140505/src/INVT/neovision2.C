/*!@file INVT/neovision2.C Neovision2 integrated demo */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/neovision2.C $
// $Id: neovision2.C 14376 2011-01-11 02:44:34Z pez $
//

#include "Image/OpenCVUtil.H" // must be first to avoid conflicting defs of int64, uint64

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/ModelParamBatch.H"
#include "Devices/DeviceOpts.H"
#include "Devices/IEEE1394grabber.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/PrefsWindow.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Layout.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "NeovisionII/Nv2LabelReader.H"
#include "NeovisionII/nv2_common.h"
#include "Neuro/NeoBrain.H"
#include "Neuro/EnvInferoTemporal.H"
#include "Neuro/EnvSaliencyMap.H"
#include "Neuro/EnvSegmenterConfigurator.H"
#include "Neuro/EnvVisualCortex.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Transport/TransportOpts.H"
#include "Util/FpsTimer.H"
#include "Util/Pause.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/SyncJobServer.H"
#include "Util/SysInfo.H"
#include "Util/TextLog.H"
#include "Util/WorkThreadServer.H"
#include "Util/csignals.H"
#include "rutz/shared_ptr.h"
#include "rutz/trace.h"

#include <ctype.h>
#include <deque>
#include <iterator>
#include <limits>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string.h>
#include <sys/resource.h>
#include <signal.h>
#include <time.h>
#include <vector>

const size_t PREFERRED_TEXT_LENGTH = 42;

// ######################################################################
class EnvSimulationViewer : public ModelComponent
{
public:
  EnvSimulationViewer(OptionManager& mgr);

  virtual ~EnvSimulationViewer();

  virtual void paramChanged(ModelParamBase* const param, const bool valueChanged, ParamClient::ChangeStatus* status);

  bool shouldQuit() const { return itsDoQuit; }

  OModelParam<Dims> itsInputDims;
  OModelParam<bool> optAsyncUi;
  OModelParam<uint> optAsyncUiQueueSize;
  OModelParam<bool> optFlushUiQueue;
  OModelParam<bool> itsNeovision2Gui;
  OModelParam<bool> optCompactDisplay;
  OModelParam<byte> optExtendedDisplay;
  OModelParam<size_t> optDispZoom;
  OModelParam<size_t> optInputReduce;
  OModelParam<std::string> optMainwinTitle;
  OModelParam<bool> itsSaveVcx;
  OModelParam<bool> itsSaveSm;

  bool itsDoQuit;
};


// ######################################################################
static const ModelOptionDef OPT_AsyncUi =
  { MODOPT_FLAG, "AsyncUi", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to run the UI asynchronously in a background thread. "
    "This will give lower latencies and better framerates on a multi-core "
    "system, but will also allow frames to be dropped in order to maintain "
    "real-time responsiveness. Use --noasync-ui when it is critical that "
    "all frames be processed through the ui queue even if it cannot be done "
    "in real time.",
    "async-ui", '\0', "", "true" };

static const ModelOptionDef OPT_AsyncUiQueueSize =
  { MODOPT_ARG(uint), "AsyncUiQueueSize", &MOC_OUTPUT, OPTEXP_CORE,
    "How many frames to hold in the asynchronous UI queue before "
    "we start dropping old frames",
    "async-ui-qsize", '\0', "<uint>", "2" };

static const ModelOptionDef OPT_FlushUiQueue =
  { MODOPT_FLAG, "FlushUiQueue", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to flush UI frames that are left in the queue "
    "when the main loop exits",
    "flush-ui-queue", '\0', "", "false" };

static const ModelOptionDef OPT_Neovision2Gui =
  { MODOPT_FLAG, "Neovision2Gui", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to show/save the neovision2 GUI",
    "neovision2-gui", '\0', "", "true" };

static const ModelOptionDef OPT_CompactDisplay =
  { MODOPT_FLAG, "EsvCompactDisplay", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to start the GUI in compact display mode",
    "compact-display", '\0', "", "true" };

static const ModelOptionDef OPT_ExtendedDisplay =
  { MODOPT_ARG(byte), "EsvExtendedDisplay", &MOC_OUTPUT, OPTEXP_CORE,
    "Amount of extra info to display in the GUI:\n"
    "    level 0: just input + saliency map\n"
    "    level 1: add meters underneath input+salmap\n"
    "    level 2: add single row of small internal maps\n"
    "    level 3: add double row of medium-sized internal maps",
    "extended-display", '\0', "<int>", "2" };

static const ModelOptionDef OPT_DispZoom =
  { MODOPT_ARG(size_t), "EsvDispZoom", &MOC_OUTPUT, OPTEXP_CORE,
    "Number of octaves to zoom in on the small maps",
    "disp-zoom", '\0', "size_t", "4" };

static const ModelOptionDef OPT_InputReduce =
  { MODOPT_ARG(size_t), "EsvInputReduce", &MOC_OUTPUT, OPTEXP_CORE,
    "Number of octaves to reduce the input by, for display purposes only",
    "input-reduce", '\0', "size_t", "0" };

static const ModelOptionDef OPT_MainwinTitle =
  { MODOPT_ARG_STRING, "MainwinTitle", &MOC_OUTPUT, OPTEXP_CORE,
    "Title to use for main output window",
    "mainwin-title", '\0', "<string>", "neovision2" };

static const ModelOptionDef OPT_SaveVcx =
  { MODOPT_FLAG, "SaveVcx", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to save the VisualCortex (VCX) output",
    "save-vcx", '\0', "", "false" };

static const ModelOptionDef OPT_SaveSm =
  { MODOPT_FLAG, "SaveSm", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to save the SaliencyMap (Sm) output",
    "save-sm", '\0', "", "false" };

// ######################################################################
EnvSimulationViewer::EnvSimulationViewer(OptionManager& mgr) :
  ModelComponent(mgr, "Embeddable Simulation Viewer", "EnvSimulationViewer"),
  itsInputDims(&OPT_InputFrameDims, this),
  optAsyncUi(&OPT_AsyncUi, this),
  optAsyncUiQueueSize(&OPT_AsyncUiQueueSize, this),
  optFlushUiQueue(&OPT_FlushUiQueue, this),
  itsNeovision2Gui(&OPT_Neovision2Gui, this),
  optCompactDisplay(&OPT_CompactDisplay, this, ALLOW_ONLINE_CHANGES),
  optExtendedDisplay(&OPT_ExtendedDisplay, this, ALLOW_ONLINE_CHANGES),
  optDispZoom(&OPT_DispZoom, this, ALLOW_ONLINE_CHANGES),
  optInputReduce(&OPT_InputReduce, this, ALLOW_ONLINE_CHANGES),
  optMainwinTitle(&OPT_MainwinTitle, this),
  itsSaveVcx(&OPT_SaveVcx, this),
  itsSaveSm(&OPT_SaveSm, this),
  itsDoQuit(false)
{ }

EnvSimulationViewer::~EnvSimulationViewer()
{ }

void EnvSimulationViewer::paramChanged(ModelParamBase* const param, const bool valueChanged,
                                       ParamClient::ChangeStatus* status)
{
  if (param == &itsInputDims) {
    const size_t excess_size = size_t(0.5 * log2(itsInputDims.getVal().sz() / 1000000.0));
    if (excess_size > optInputReduce.getVal()) optInputReduce.setVal(excess_size);
  } else if (param == &optInputReduce) {
    const size_t val = optInputReduce.getVal();
    const Dims d = itsInputDims.getVal();

    if (val > 16) *status = ParamClient::CHANGE_REJECTED;
    else if (d.isNonEmpty() && val > 0 && ((d.w() / (1 << val)) < 32 || (d.h() / (1 << val)) < 32))
      *status = ParamClient::CHANGE_REJECTED;
  }
}

// ######################################################################
struct Nv2UiData
{
  Nv2UiData(const int map_zoom_) :
    accepted_training_label(),
    remote_command(),
    map_zoom(map_zoom_),
    targetLoc(-1,-1),
    ncpu(numCpus()),
    text_log_file("")
  { }

  FpsTimer::State time_state;
  std::string accepted_training_label;
  std::string remote_command;
  const int map_zoom;
  Point2D<int> targetLoc;
  const int ncpu;
  std::string text_log_file;
};

// ######################################################################
class Nv2UiJob : public JobServer::Job
{
public:
  Nv2UiJob(OutputFrameSeries* ofs_,
           EnvSimulationViewer* esv_,
           EnvInferoTemporal* eit_,
           const Nv2UiData& uidata_,
           EnvSaliencyMap* sm_,
           EnvSegmenter* ese_,
           NeoBrain* nb_,
           Image<PixRGB<byte> > rgbin_,
           Image<byte> vcxmap_
           , Image<byte> Imap_
           , Image<byte> Cmap_
           , Image<byte> Omap_
#ifdef ENV_WITH_DYNAMIC_CHANNELS
           , Image<byte> Fmap_
           , Image<byte> Mmap_
#endif
           ) :
    ofs(ofs_), esv(esv_), eit(eit_), uidata(uidata_), sm(sm_), ese(ese_), neoBrain(nb_), rgbin(rgbin_),
    vcxmap(vcxmap_), Imap(Imap_), Cmap(Cmap_), Omap(Omap_),
#ifdef ENV_WITH_DYNAMIC_CHANNELS
    Fmap(Fmap_), Mmap(Mmap_),
#endif
    m_dispzoom(1 << esv->optDispZoom.getVal()), m_inputreduce(esv->optInputReduce.getVal())
  { }

  // ####################
  unsigned int getHalfZoom() const
  {
    const int div = (esv->optExtendedDisplay.getVal() >= 4 && esv->optExtendedDisplay.getVal() < 10) ? 2 : 4;
    return std::max(size_t(1), m_dispzoom/div);
  }

  // ####################
  Layout<PixRGB<byte> > makeInputMarkup(const Rectangle& foa, const Image<byte>& foamask,
                                        const EnvSaliencyMap::State& smstate, const uint32_t patch_id) const
  {
    Image<PixRGB<byte> > markup = rgbin;

    if (foa.isValid()) drawRectSquareCorners(markup, foa, PixRGB<byte>(255, 255, 0), 3 << m_inputreduce);

    if (uidata.targetLoc.isValid())
      drawCircle(markup, uidata.targetLoc, 3, PixRGB<byte>(60, 220, 255), 3 << m_inputreduce);

    // draw the first most salient loc:
    drawRectSquareCorners(markup,
                          Rectangle(smstate.fullres_maxpos - uidata.map_zoom/2, Dims(uidata.map_zoom, uidata.map_zoom)),
                          PixRGB<byte>(255, 0, 0), 3 << m_inputreduce);

    // draw the next n most salient loc:
    for (uint i = 1; i < smstate.nMostSalientLoc.size(); ++i)
      {
        const EnvSaliencyMap::LocInfo locInfo = smstate.nMostSalientLoc[i];
        drawRectSquareCorners(markup, Rectangle(locInfo.fullres_maxpos - uidata.map_zoom/2,
                                                Dims(uidata.map_zoom, uidata.map_zoom)),
                              PixRGB<byte>(150, 0, 0), 3 << m_inputreduce);
      }

    for (size_t i = 0; i < m_inputreduce; ++i) markup = decXY(markup);

    if (foamask.initialized()) drawContour2D(rescaleNI(foamask, markup.getDims()), markup, PixRGB<byte>(0,255,0), 2);

    const std::string lines[2] = {
      sformat("peak %3d in %3dx%3d foa @ (%3d,%3d)", int(smstate.maxval),
              foa.isValid() ? foa.width() : -1, foa.isValid() ? foa.height() : -1,
              smstate.fullres_maxpos.i, smstate.fullres_maxpos.j),
        sformat("%s #%06u [%5.2ffps, %5.1f%%CPU]", convertToString(uidata.time_state.elapsed_time).c_str(),
                (unsigned int) patch_id, uidata.time_state.recent_fps, uidata.time_state.recent_cpu_usage*100.0)
    };

    const Image<PixRGB<byte> > textarea =
      makeMultilineTextBox(markup.getWidth(), &lines[0], 2,
                           PixRGB<byte>(255, 255, 0), PixRGB<byte>(0,0,0), PREFERRED_TEXT_LENGTH);

    return vcat(markup, textarea);
  }

  // ####################
  Layout<PixRGB<byte> >
  makeSalmapMarkup(const EnvSaliencyMap::State& smstate) const
  {
    Image<PixRGB<byte> > zoomedsm = zoomXY(smstate.salmap, m_dispzoom, m_dispzoom);

    // draw the first most salient loc:
    drawRectSquareCorners(zoomedsm, Rectangle(smstate.lowres_maxpos * m_dispzoom, Dims(m_dispzoom, m_dispzoom)),
                          PixRGB<byte>(255, 0, 0), 3);

    // draw the next n most salient locs:
    for (uint i = 1; i < smstate.nMostSalientLoc.size(); ++i)
      {
        const EnvSaliencyMap::LocInfo locInfo = smstate.nMostSalientLoc[i];
        drawRectSquareCorners(zoomedsm, Rectangle(locInfo.lowres_maxpos * m_dispzoom, Dims(m_dispzoom, m_dispzoom)),
                              PixRGB<byte>(150, 0, 0), 3);
      }

    const std::string valstring = sformat("%d", int(smstate.maxval));

    const SimpleFont font = SimpleFont::fixedMaxWidth(zoomedsm.getWidth() / 30);

    Point2D<int> textpos = smstate.lowres_maxpos * m_dispzoom;
    textpos.j -= font.h() + 2; if (textpos.j < 0) textpos.j += m_dispzoom + 2;

    writeText(zoomedsm, textpos, valstring.c_str(),
              PixRGB<byte>(255, 0, 0), PixRGB<byte>(0, 0, 0), font, true);

#ifdef VSS_DEMO
    Image<PixRGB<byte> > histo =
      neoBrain->getSaliencyHisto(Dims(zoomedsm.getWidth(), 62), PixRGB<byte>(0,0,0), PixRGB<byte>(180,180,180));
    return vcat(zoomedsm, histo);
#else
    return zoomedsm;
#endif
  }

  // ####################
  Layout<PixRGB<byte> > makeCmapsMarkup() const
  {
    unsigned int halfzoom = this->getHalfZoom();
    if (esv->optExtendedDisplay.getVal() == 10) halfzoom /= 2;

    Image<PixRGB<byte> > cmaps[] = {
      zoomXY(Imap, halfzoom, halfzoom),
      zoomXY(Cmap, halfzoom, halfzoom),
      zoomXY(Omap, halfzoom, halfzoom),
#ifdef ENV_WITH_DYNAMIC_CHANNELS
      zoomXY(Fmap, halfzoom, halfzoom),
      zoomXY(Mmap, halfzoom, halfzoom),
#endif
      zoomXY(vcxmap, halfzoom, halfzoom)
    };

    const char* labels[] = {
      "I", "C", "O",
#ifdef ENV_WITH_DYNAMIC_CHANNELS
      "F", "M",
#endif
      "VC"
    };

    for (size_t i = 0; i < sizeof(labels) / sizeof(labels[0]); ++i) {
      const SimpleFont font = SimpleFont::fixedMaxWidth(cmaps[i].getWidth() / 20);
      writeText(cmaps[i], Point2D<int>(1,1), labels[i], PixRGB<byte>(0), PixRGB<byte>(255), font);
      drawLine(cmaps[i], Point2D<int>(0,0), Point2D<int>(cmaps[i].getWidth()-1,0), PixRGB<byte>(255), 1);
      drawLine(cmaps[i], Point2D<int>(0,0), Point2D<int>(0,cmaps[i].getHeight()-1), PixRGB<byte>(255), 1);
    }

    const size_t nrows = esv->optExtendedDisplay.getVal() >= 4 ? 2 : 1;

    return arrcat(&cmaps[0], sizeof(cmaps) / sizeof(cmaps[0]), (sizeof(cmaps) / sizeof(cmaps[0]) + (nrows-1)) / nrows);
  }

  // ####################
  Image<PixRGB<byte> > makeInhibitionMarkup() const
  {
    Image<byte> inh = sm->getInhibmap();
    if (!inh.initialized()) inh = Image<byte>(vcxmap.getDims(), ZEROS);

    Image<byte> inr = Image<byte>(sm->getInertiaMap());
    if (!inr.initialized()) inr = Image<byte>(vcxmap.getDims(), ZEROS);

    Image<PixRGB<byte> > rgb(vcxmap.getDims(), NO_INIT);
    Image<PixRGB<byte> >::iterator aptr = rgb.beginw();
    Image<PixRGB<byte> >::iterator stop = rgb.endw();

    Image<byte>::const_iterator rptr = inh.begin();
    Image<byte>::const_iterator gptr = inr.begin();

    while (aptr != stop) *aptr++ = PixRGB<byte>(*rptr++, *gptr++, 0);

    return zoomXY(rgb, esv->optExtendedDisplay.getVal() == 10 ? getHalfZoom() / 2 : getHalfZoom());
  }

  // ####################
  Image<PixRGB<byte> > makeMeters(const size_t nx, const Dims& meterdims) const
  {
    const double maxcpu = uidata.ncpu <= 0 ? 100.0 : uidata.ncpu * 100.0;

    const double nothresh = std::numeric_limits<double>::max();

    const MeterInfo infos[] = {
      { "dvcx/dt", sm->getVcxFlicker(), 1.0, nothresh, PixRGB<byte>(0, 255, 0) },
      { "dfactor", sm->getDynamicFactor(), 1.0, nothresh, PixRGB<byte>(128, 0, 255) },
#ifdef VSS_DEMO
      { "boringness", neoBrain->getBoringness(), 128.0, nothresh, PixRGB<byte>(192, 255, 0) },
      { "excitement", neoBrain->getExcitementLevel(), 256.0, nothresh, PixRGB<byte>(255, 0, 32) },
      { "sleepiness", neoBrain->getSleepLevel(), 1000.0, nothresh, PixRGB<byte>(255, 0, 32) },
#endif
      { "confidence", eit->getMaxConfidence(), 1.0, eit->getConfidenceThresh(), PixRGB<byte>(0, 255, 128) },
      { "cpu%", uidata.time_state.recent_cpu_usage*100.0, maxcpu, nothresh, PixRGB<byte>(255, 165, 0) },
      { "fps", uidata.time_state.recent_fps, 60.0, nothresh, PixRGB<byte>(0, 128, 255) }
    };

    return drawMeters(&infos[0], sizeof(infos) / sizeof(infos[0]), nx, meterdims);
  }

  // ####################
  virtual void run()
  {
    Point2D<int> scaled_maxpos(-1,-1);

    const nub::soft_ref<ImageDisplayStream> ids = ofs->findFrameDestType<ImageDisplayStream>();

    const rutz::shared_ptr<XWinManaged> uiwin = ids.is_valid() ?
      ids->getWindow(esv->optCompactDisplay.getVal() ? esv->optMainwinTitle.getVal() : "input")
      : rutz::shared_ptr<XWinManaged>();

    Point2D<int> forceTrackLocation(-1,-1);

    if (uiwin.is_valid()) {
      XButtonEvent ev;
      if (uiwin->getLastButtonEvent(&ev) && ev.button == 1) forceTrackLocation = Point2D<int>(ev.x, ev.y);
     }

    if (forceTrackLocation.isValid()) {
      const Point2D<int> candidate = forceTrackLocation * (1 << m_inputreduce) + (1 << m_inputreduce) / 2;

      if (rgbin.coordsOk(candidate)) {
        scaled_maxpos = candidate;
        neoBrain->setTarget(scaled_maxpos, rgbin, -1, false, true);
        //neoBrain->setKeepTracking(true);
      }
    } else if (uidata.targetLoc.isValid()) {
      scaled_maxpos = uidata.targetLoc;
      ASSERT(rgbin.coordsOk(scaled_maxpos));
    }

    const EnvSaliencyMap::State smstate = sm->getSalmap(vcxmap, scaled_maxpos);

#ifdef VSS_DEMO
    neoBrain->updateBoringness(smstate.salmap, smstate.maxval);
    neoBrain->updateExcitement(sm->getVcxFlicker());
#endif

    Image<byte> foamask;
    Image<PixRGB<byte> > segmentdisp;

    // Send the first most salient locations to be identified
    const Rectangle foa = ese->getFoa(rgbin, smstate.fullres_maxpos, &foamask, &segmentdisp);

    //if (foa.isValid()) {
    //  const Point2D<int> objCenter = Point2D<int>(foa.topLeft().i + foa.width()/2, foa.topLeft().j + foa.height()/2);
    //  neoBrain->setTarget(objCenter, rgbin, smstate.maxval);
    //} else if (!uidata.targetLoc.isValid())
    //  neoBrain->setTarget(smstate.fullres_maxpos, rgbin, smstate.maxval);
    if (!uidata.targetLoc.isValid())
            neoBrain->setTarget(smstate.fullres_maxpos, rgbin, smstate.maxval);

    const uint32_t patch_id = uidata.time_state.frame_number;

    eit->sendPatch(patch_id, rgbin, foa,
                   uidata.time_state.elapsed_time,
                   uidata.accepted_training_label.length() > 0,
                   uidata.accepted_training_label,
                   uidata.remote_command,
                   smstate.fullres_maxpos);

    // Send the next N most salient locations to be identified:
    for (uint i = 1; i < smstate.nMostSalientLoc.size(); ++i) {
      Image<byte> nextFoamask;
      Image<PixRGB<byte> > nextSegmentdisp;

      const EnvSaliencyMap::LocInfo locInfo = smstate.nMostSalientLoc[i];
      const Rectangle nextFoa = ese->getFoa(rgbin, locInfo.fullres_maxpos, &nextFoamask, &nextSegmentdisp);

      eit->sendPatch(patch_id, rgbin, nextFoa,
                     uidata.time_state.elapsed_time,
                     uidata.accepted_training_label.length() > 0,
                     uidata.accepted_training_label,
                     uidata.remote_command,
                     locInfo.fullres_maxpos);
    }

    // log various bits of info (these calls will do nothing if the log filename is empty):
    textLog(uidata.text_log_file, "FOAbox", convertToString(foa));

    const FrameState os = ofs->updateNext();

    // save maps if requested:
    if (esv->itsSaveVcx.getVal()) ofs->writeGray(vcxmap, "VCO", FrameInfo("VisualCortex output map", SRC_POS));

    if (esv->itsSaveSm.getVal()) ofs->writeGray(smstate.salmap, "SM", FrameInfo("SaliencyMap output map", SRC_POS));

    // ##### compact displays
    if (esv->itsNeovision2Gui.getVal() && esv->optCompactDisplay.getVal()) {
      ofs->closeStream("input");
      ofs->closeStream("salmap");
      ofs->closeStream("cmaps");
      ofs->closeStream("inhibition,inertia");
      ofs->closeStream("segmentation");
      ofs->closeStream("meters");

      Layout<PixRGB<byte> > img;

      // ##### mode 10 displays (for display on HD monitor)
      if (esv->optExtendedDisplay.getVal() == 10) {
        // let's start with an HD display of the input + markups:
        Image<PixRGB<byte> > markup = rgbin;
        if (foa.isValid()) drawRectSquareCorners(markup, foa, PixRGB<byte>(255, 255, 0), 3 << m_inputreduce);

        if (uidata.targetLoc.isValid())
          drawCircle(markup, uidata.targetLoc, 3, PixRGB<byte>(60, 220, 255), 3 << m_inputreduce);

        // draw the first most salient loc
        drawRectSquareCorners(markup,
                              Rectangle(smstate.fullres_maxpos - uidata.map_zoom/2,
                                        Dims(uidata.map_zoom, uidata.map_zoom)),
                              PixRGB<byte>(255, 0, 0), 3 << m_inputreduce);

        // draw the next n most salient locs:
        for (uint i = 1; i < smstate.nMostSalientLoc.size(); ++i) {
          const EnvSaliencyMap::LocInfo locInfo = smstate.nMostSalientLoc[i];
          drawRectSquareCorners(markup,
                                Rectangle(locInfo.fullres_maxpos - uidata.map_zoom/2,
                                 Dims(uidata.map_zoom, uidata.map_zoom)), PixRGB<byte>(150, 0, 0), 3 << m_inputreduce);
        }

        for (size_t i = 0; i < m_inputreduce; ++i) markup = decXY(markup);

        if (foamask.initialized()) drawContour2D(rescaleNI(foamask,markup.getDims()), markup, PixRGB<byte>(0,255,0), 2);

        // that's it for this window, let's send it out to display:
        ofs->writeRGB(markup, esv->optMainwinTitle.getVal(), FrameInfo("copy of input", SRC_POS));

        // the salmap:
        const unsigned int halfzoom = 8;
        Image<PixRGB<byte> > zoomedsm = zoomXY(smstate.salmap, halfzoom, halfzoom);

        drawRectSquareCorners(zoomedsm, Rectangle(smstate.lowres_maxpos * halfzoom, Dims(halfzoom, halfzoom)),
                              PixRGB<byte>(255, 0, 0), 3);

        const std::string valstring = sformat("%d", int(smstate.maxval));
        const SimpleFont font = SimpleFont::fixedMaxWidth(zoomedsm.getWidth() / 30);
        Point2D<int> textpos = smstate.lowres_maxpos * halfzoom;
        textpos.j -= font.h() + 2; if (textpos.j < 0) textpos.j += halfzoom + 2;
        writeText(zoomedsm, textpos, valstring.c_str(), PixRGB<byte>(255, 0, 0), PixRGB<byte>(0, 0, 0), font, true);

        Image<PixRGB<byte> > inh = this->makeInhibitionMarkup();
        drawLine(inh, Point2D<int>(0,0), Point2D<int>(inh.getWidth()-1,0), PixRGB<byte>(255, 255, 255), 1);
        drawLine(inh, Point2D<int>(0,0), Point2D<int>(0,inh.getHeight()-1), PixRGB<byte>(255, 255, 255), 1);

        if (!segmentdisp.initialized())
          segmentdisp = Image<PixRGB<byte> >(inh.getDims(), ZEROS);
        else {
          segmentdisp = rescaleNI(segmentdisp, inh.getDims());
          drawContour2D(rescaleNI(foamask, inh.getDims()), segmentdisp, PixRGB<byte>(0,255,0), 2);
        }
        drawLine(segmentdisp, Point2D<int>(0,0), Point2D<int>(inh.getWidth()-1,0),
                 PixRGB<byte>(255, 255, 255), 1);
        drawLine(segmentdisp, Point2D<int>(0,0), Point2D<int>(0,inh.getHeight()-1),
                 PixRGB<byte>(255, 255, 255), 1);
        Layout<PixRGB<byte> > inl = vcat(inh, segmentdisp);

        img = vcat(zoomedsm, this->makeMeters(2, Dims(zoomedsm.getDims().w() / 2, 13)));

        // now some info:
        const std::string lines[1] =
          {
            sformat("peak %3d in %3dx%3d foa @ (%4d,%4d) %04dx%04d %s #%06u [%3.2ffps, %4.1f%%CPU]",
                    int(smstate.maxval),
                    foa.isValid() ? foa.width() : -1,
                    foa.isValid() ? foa.height() : -1,
                    smstate.fullres_maxpos.i,
                    smstate.fullres_maxpos.j,
                    rgbin.getWidth(), rgbin.getHeight(),
                    convertToString(uidata.time_state.elapsed_time).c_str(),
                    (unsigned int) patch_id,
                    uidata.time_state.recent_fps,
                    uidata.time_state.recent_cpu_usage*100.0)
          };

        const Image<PixRGB<byte> > textarea =
          makeMultilineTextBox(img.getWidth(), &lines[0], 1, PixRGB<byte>(255, 255, 0), PixRGB<byte>(0,0,0),
                               PREFERRED_TEXT_LENGTH, 10);
        img = vcat(img, textarea);

        // now the cmaps and friends:
        const Layout<PixRGB<byte> > cmaps = this->makeCmapsMarkup();
        inl = hcat(cmaps, inl);
        img = vcat(img, inl);

        ofs->writeRgbLayout(img, "control-panel", FrameInfo("copy of input", SRC_POS));
        ////////////////////////////////
      } else {
        img = hcat(this->makeInputMarkup(foa, foamask, smstate, patch_id), this->makeSalmapMarkup(smstate));

        if (esv->optExtendedDisplay.getVal() > 0) {
          // ##### mode 2+ displays
          if (esv->optExtendedDisplay.getVal() >= 2)
            {
              const Layout<PixRGB<byte> > cmaps = this->makeCmapsMarkup();
              Image<PixRGB<byte> > inh = this->makeInhibitionMarkup();
              drawLine(inh, Point2D<int>(0,0), Point2D<int>(inh.getWidth()-1,0), PixRGB<byte>(255, 255, 255), 1);
              drawLine(inh, Point2D<int>(0,0), Point2D<int>(0,inh.getHeight()-1), PixRGB<byte>(255, 255, 255), 1);

              const SimpleFont font = SimpleFont::fixedMaxWidth(inh.getWidth() / 20);
              writeText(inh, Point2D<int>(1,1), "INH", PixRGB<byte>(0), PixRGB<byte>(255), font);

              if (!segmentdisp.initialized())
                segmentdisp = Image<PixRGB<byte> >(inh.getDims(), ZEROS);
              else {
                segmentdisp = rescaleNI(segmentdisp, inh.getDims());

                drawContour2D(rescaleNI(foamask, inh.getDims()), segmentdisp, PixRGB<byte>(0,255,0), 2);
                writeText(segmentdisp, Point2D<int>(1,1), "SEG", PixRGB<byte>(0), PixRGB<byte>(255), font);
              }
              drawLine(segmentdisp, Point2D<int>(0,0), Point2D<int>(inh.getWidth()-1,0),
                       PixRGB<byte>(255, 255, 255), 1);
              drawLine(segmentdisp, Point2D<int>(0,0), Point2D<int>(0,inh.getHeight()-1),
                       PixRGB<byte>(255, 255, 255), 1);

              if (esv->optExtendedDisplay.getVal() >= 4)
                img = vcat(img, hcat(cmaps, vcat(inh,segmentdisp)));
              else
                img = vcat(img, hcat(cmaps, hcat(inh,segmentdisp)));
            }

          img = vcat(img, this->makeMeters(2, Dims(img.getDims().w() / 2, 13)));

          // ##### mode 3+ displays
          if (esv->optExtendedDisplay.getVal() >= 3)
            {
              NeoBrain::Stats stats = neoBrain->getStats();

              const std::string lines[5] = {
                sformat(" track error #frames %04lu (%04lu) | target track #frames %04lu (%04lu)",
                        stats.bigerrframes, stats.bigerrframes_thresh, stats.targetframes, stats.targetframes_thresh),
                sformat("  non-motion #frames %04lu (%04lu) |   stop track #frames %04lu (%04lu)",
                        stats.nomoveframes, stats.nomoveframes_thresh, stats.stopframes, stats.stopframes_thresh),
                "",
                sformat("    track err = %.3f = %.3f+%.3f (%.3f, %.3f)",
                        stats.last_err, stats.last_xerr, stats.last_yerr, stats.err_tol, stats.dist_tol),
                sformat("    leftEyeTilt = %.3f leftEyePan = %.3f headPan = %.3f",
                        stats.leftEyeTiltPos, stats.leftEyePanPos, stats.headPanPos)
              };

              const Image<PixRGB<byte> > textarea =
                makeMultilineTextBox(img.getDims().w(), &lines[0], 5,
                                     PixRGB<byte>(128, 180, 255), PixRGB<byte>(0,0,0), 150);
              img = vcat(img, textarea);
            }
        }
        ofs->writeRgbLayout(img, esv->optMainwinTitle.getVal(), FrameInfo("copy of input", SRC_POS));
      }
    } else if (esv->itsNeovision2Gui.getVal()) {
      // ##### multiwindow displays
      ofs->closeStream(esv->optMainwinTitle.getVal());

      ofs->writeRgbLayout(this->makeInputMarkup(foa, foamask, smstate, patch_id),
                          "input", FrameInfo("copy of input", SRC_POS));

      ofs->writeRgbLayout(this->makeSalmapMarkup(smstate), "salmap", FrameInfo("saliency map", SRC_POS));

      if (esv->optExtendedDisplay.getVal()) {
        ofs->writeRgbLayout(this->makeCmapsMarkup(), "cmaps", FrameInfo("channel conspicuity maps", SRC_POS));

        ofs->writeRGB(this->makeInhibitionMarkup(), "inhibition,inertia",
                      FrameInfo("inertia and inhibition of return", SRC_POS));

        ofs->writeRGB(this->makeMeters(1, Dims(400, 13)), "meters");
      } else {
        ofs->closeStream("cmaps");
        ofs->closeStream("inhibition,inertia");
        ofs->closeStream("meters");
      }

      if (segmentdisp.initialized()) {
        segmentdisp = zoomXY(segmentdisp, m_dispzoom);
        drawContour2D(zoomXY(foamask, m_dispzoom), segmentdisp, PixRGB<byte>(0,255,0), 2);
        ofs->writeRGB(segmentdisp, "segmentation");
      } else ofs->closeStream("segmentation");
    }

    std::vector<Nv2LabelReader::LabeledImage> images = eit->getLabeledImages(PREFERRED_TEXT_LENGTH);

    for (size_t i = 0; i < images.size(); ++i)
      {
        ofs->writeRGB(images[i].img, images[i].ident, FrameInfo("object-labeled image", SRC_POS));
#ifdef VSS_DEMO
        neoBrain->sayObjectLabel(images[i].label, /*confidence = */ 0, true);
#endif
      }

    if (os == FRAME_FINAL) esv->itsDoQuit = true;
  }

  // ####################
  virtual const char* jobType() const { return "Nv2UiJob"; }

private:
  OutputFrameSeries* const ofs;
  EnvSimulationViewer* const esv;
  EnvInferoTemporal* const eit;
  const Nv2UiData uidata;
  EnvSaliencyMap* const sm;
  EnvSegmenter* const ese;
  NeoBrain* const neoBrain;
  Image<PixRGB<byte> > rgbin;
  const Image<byte> vcxmap;
  const Image<byte> Imap;
  const Image<byte> Cmap;
  const Image<byte> Omap;
#ifdef ENV_WITH_DYNAMIC_CHANNELS
  const Image<byte> Fmap;
  const Image<byte> Mmap;
#endif
  const size_t m_dispzoom;
  const size_t m_inputreduce;
};

// ######################################################################
static const ModelOptionDef OPT_WithObjrecMode =
  { MODOPT_FLAG, "WithObjrecMode", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to include an 'objrec' mode which toggles parameters "
    "to values suitable for object recognition training.",
    "with-objrec-mode", '\0', "", "false" };

static const ModelOptionDef OPT_ALIASheadDemo =
  { MODOPT_ALIAS, "ALIASheadDemo", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the head demo",
    "head-demo", '\0', "",
    "--in=V4L2 "
    "--framegrabber-dims=320x240 "
    "--framegrabber-mode=RGB24 "
    "--framegrabber-chan=0 "
    "--framegrabber-dev=/dev/video1 "
    "--evc-type=I:0.75C:0.75O:0.75F:0.75M:1.0 "
    "--mainwin-title=Sal "
    "--neobrain-speak-saliency "
    "--neobrain-speak-objects "
    "--with-objrec-mode "
  };

static const ModelOptionDef OPT_ALIASheadDemo21 =
  { MODOPT_ALIAS, "ALIASheadDemo21", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the head demo on ilab21",
    "head-demo21", '\0', "",
    "--in=v4l2 "
    "--framegrabber-dims=720x480 "
    "--framegrabber-mode=HM12 "
    "--framegrabber-chan=2 "
    "--framegrabber-dev=/dev/video32 "
    "--evc-type=I:0.75C:0.75O:0.75F:0.75M:1.0 "
    "--mainwin-title=Sal "
    "--with-objrec-mode "
    "--neobrain-speak-saliency "
    "--evc-multithreaded "
  };

static const ModelOptionDef OPT_ALIASheadDemo3 =
  { MODOPT_ALIAS, "ALIASheadDemo3", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the head demo on ilab3",
    "head-demo3", '\0', "",
    "--in=v4l2 "
    "--evc-type=I:0.75C:0.75O:0.75F:0.75M:1.0 "
    "--mainwin-title=Sal "
    "--with-objrec-mode "
    "--neobrain-speak-saliency "
    "--evc-multithreaded "
  };

static const ModelOptionDef OPT_ALIASflea2Demo =
  { MODOPT_ALIAS, "ALIASflea2Demo", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the Flea2 demo",
    "flea2-demo", '\0', "",
    "--in=dc1394v2 "
    "--framegrabber-dims=1024x768 "
    "--framegrabber-mode=RGB24 "
    "--framegrabber-fps=15 "
    "--evc-multithreaded "
    "--noneobrain-speech "
    "--disp-zoom=4 "
  };

static const ModelOptionDef OPT_ALIASHDDemo =
  { MODOPT_ALIAS, "ALIASHDDemo", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the hd camera on ilab24",
    "hd-demo", '\0', "",
    "--in=XC "
    "--framegrabber-dims=1024x768 "
    "--patch-reader=192.168.0.229:9930 "
    "--disp-zoom=3 "
    "--evc-multithreaded "
  };

// ######################################################################
int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  signal(SIGPIPE, SIG_IGN);
  catchsignals(&signum);

  // Instantiate our various ModelComponents:

  ModelManager manager("Nv2");

  OModelParam<bool> optWithObjrecMode(&OPT_WithObjrecMode, &manager);
  OModelParam<std::string> optTextLogFile(&OPT_TextLogFile, &manager);

  nub::ref<EnvSimulationViewer> esv(new EnvSimulationViewer(manager));
  manager.addSubComponent(esv);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(manager));
  manager.addSubComponent(evc);

  nub::ref<EnvSaliencyMap> esm(new EnvSaliencyMap(manager));
  manager.addSubComponent(esm);

  nub::ref<EnvSegmenterConfigurator> esec(new EnvSegmenterConfigurator(manager));
  manager.addSubComponent(esec);

  nub::ref<EnvInferoTemporal> eit(new EnvInferoTemporal(manager));
  manager.addSubComponent(eit);

  nub::ref<NeoBrain> neoBrain(new NeoBrain(manager));
  manager.addSubComponent(neoBrain);

  manager.requestOptionAlias(&OPT_ALIASheadDemo);
  manager.requestOptionAlias(&OPT_ALIASheadDemo3);
  manager.requestOptionAlias(&OPT_ALIASheadDemo21);
  manager.requestOptionAlias(&OPT_ALIASflea2Demo);
  manager.requestOptionAlias(&OPT_ALIASHDDemo);

  manager.exportOptions(MC_RECURSE);

#if defined(HAVE_IEEE1394)
  // input comes from firewire camera 640x480/rgb/15fps by default
  manager.setOptionValString(&OPT_InputFrameSource, "ieee1394");
  manager.setOptionValString(&OPT_FrameGrabberMode, "RGB24");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "15");
#elif defined(HAVE_QUICKTIME_QUICKTIME_H)
  manager.setOptionValString(&OPT_InputFrameSource, "qtgrab");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
#endif

  // output goes to the screen by default
  manager.setOptionValString(&OPT_OutputFrameSink, "display");

  // change some default values
  manager.setOptionValString(&OPT_EvcColorSmoothing, "true");
  manager.setOptionValString(&OPT_EsmInertiaHalfLife, "60");
  manager.setOptionValString(&OPT_EsmIorStrength, "8.0");

  if (manager.parseCommandLine(argc, argv, "<ip1:port1,ip2:port2,...>", 0, 1) == false) return(1);

  eit->initReaders(manager.numExtraArgs() > 0 ? manager.getExtraArg(0) : "");

  manager.start();

  neoBrain->init(ifs->peekDims());

  Nv2UiData uidata(1 << evc->getMapLevel());
  uidata.text_log_file = optTextLogFile.getVal();

  PrefsWindow pwin("control panel", SimpleFont::FIXED(8));
  pwin.setValueNumChars(16);

  pwin.addPrefsForComponent(esv.get());
  pwin.addPrefsForComponent(esm.get());
  pwin.addPrefsForComponent(esec->getSeg().get());
  pwin.addPrefsForComponent(eit.get());
  pwin.addPrefsForComponent(neoBrain.get(), true);
  pwin.addPrefsForComponent(evc.get());

  PrefItemBln prefPause(&pwin, "pause", false);
  PrefItemStr prefRemoteCommand(&pwin, "remote command", uidata.remote_command);
  PrefItemBln prefInTrainingMode(&pwin, "in training mode", false);
  PrefItemBln prefInObjRecMode(optWithObjrecMode.getVal() ? &pwin : 0, "in ObjRec mode", false);
  PrefItemBln prefDoGrabFrame(&pwin, "grab frame", true);
  PrefItemBln prefCommitTrainingImage(&pwin, "commit training image", false);
  PrefItemBln prefCommitTrainingImageConfirm(&pwin, "confirm commit ??", false);
  PrefItemStr prefTrainingLabel(&pwin, "training label", "");
  PrefItemByt prefFontSize(&pwin, "font size", 6);

  PrefsWindow inputprefs;
  inputprefs.addPrefsForComponent(ifs->getFrameSource().get());
  pwin.setFont(SimpleFont::fixedMaxWidth(prefFontSize.get()));
  inputprefs.setFont(SimpleFont::fixedMaxWidth(prefFontSize.get()));

  PauseWaiter p;

  int retval = 0;

  rutz::shared_ptr<JobServer> uiq;
  if (esv->optAsyncUi.getVal()) {
    // set up a background job server with one worker thread to
    // handle the ui jobs:
    rutz::shared_ptr<WorkThreadServer> tsrv(new WorkThreadServer("neovision2-ui", 1));

    // keep max latency low, and if we get bogged down, then drop
    // old frames rather than new ones
    tsrv->setMaxQueueSize(esv->optAsyncUiQueueSize.getVal());
    tsrv->setDropPolicy(WorkThreadServer::DROP_OLDEST);
    tsrv->setFlushBeforeStopping(esv->optFlushUiQueue.getVal());
    uiq = tsrv;
  } else {
    // set up a synchronous job server that will run all ui jobs in
    // the foreground
    uiq.reset(new SyncJobServer);
  }

  ASSERT(uiq.get() != 0);

  ifs->startStream();

  // const GenericFrameSpec fspec = ifs->peekFrameSpec();

  FpsTimer fps_timer;

  bool previous_training_mode = prefInTrainingMode.get();
  bool previous_do_fixed = esm->getUseFixed();
  Image<PixRGB<byte> > rgbin_last;

  ModelParamBatch objrecParams;
  objrecParams.addParamValue("EseDynamicFoa", false);
  objrecParams.addParamValue("EseFoaSize", 80);
  objrecParams.addParamValue("NeobrainBoringnessThresh", 2000);
  objrecParams.addParamValue("NeobrainTargetFramesThresh", (unsigned long) 2000);
  objrecParams.addParamValue("NeobrainNoMoveFramesThresh", (unsigned long) 2000);

  bool previous_objrec_mode = prefInObjRecMode.get();

  while (true)
    {
      if (signum != 0) {
          LINFO("quitting because %s was caught", signame(signum));
          retval = -1;
          break;
      }

      if (ofs->becameVoid()) {
        LINFO("quitting because output stream was closed or became void");
        break;
      }

      if (esv->shouldQuit()) break;

      //
      // update preferences window and uidata
      //

      pwin.update(); // handle pending preference window events

      setPause(prefPause.get());
      uidata.remote_command = prefRemoteCommand.get();

      prefCommitTrainingImage.setDisabled(!prefInTrainingMode.get());
      prefCommitTrainingImageConfirm.setDisabled(!prefInTrainingMode.get());
      prefTrainingLabel.setDisabled(!prefInTrainingMode.get());

      pwin.setFont(SimpleFont::fixedMaxWidth(prefFontSize.get()));

      inputprefs.setFont(SimpleFont::fixedMaxWidth(prefFontSize.get()));
      inputprefs.update();

      if (prefInObjRecMode.get()) {
        if (!previous_objrec_mode) objrecParams.installValues(&manager); // save previous values
      } else {
        if (previous_objrec_mode) objrecParams.restoreValues(&manager); //restore values
      }

      previous_objrec_mode = prefInObjRecMode.get();

      // This code enforces the "training mode" logic
      //  .. i.e., certain combinations of preferences are not possible.
      uidata.accepted_training_label = "";

      if (prefInTrainingMode.get()) {
        if (!previous_training_mode) previous_do_fixed = esm->getUseFixed();

          esm->setUseFixed(true);

          if (prefCommitTrainingImageConfirm.get()) {
            if (!prefCommitTrainingImage.get())
              prefCommitTrainingImageConfirm.set(false);
            else if (prefTrainingLabel.get().length() <= 3) {
              prefCommitTrainingImage.set(false);
              prefCommitTrainingImageConfirm.set(false);
              prefTrainingLabel.set("");

              LERROR("invalid training label %s (too short)", prefTrainingLabel.get().c_str());
            } else {
              // OK, we accept the training label as a valid one
              // and send it off to the labelers:
              uidata.accepted_training_label = prefTrainingLabel.get();
            }
          }
      } else {
        // training mode is off, certain settings not possible
        prefDoGrabFrame.set(true);
        prefCommitTrainingImage.set(false);
        prefCommitTrainingImageConfirm.set(false);
        prefTrainingLabel.set("");

        // this just handles unfixing window when training is first toggled off
        if (previous_training_mode) esm->setUseFixed(previous_do_fixed);
      }

      previous_training_mode = prefInTrainingMode.get();

      if (p.checkPause()) continue;

      //
      // get the next frame from our input source
      //

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE) break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized()) break;

      // only read in from camera if do_grab_frame
      const Image<PixRGB<byte> > rgbin = prefDoGrabFrame.get() ? input.asRgb() : rgbin_last;

      rgbin_last = rgbin;

      if (eit->belowConfidenceThresh()) uidata.targetLoc = neoBrain->trackObject(rgbin);
      else uidata.targetLoc = Point2D<int>(-1,-1);

      //
      // send the frame to the EnvVisualCortex and get the vcx output
      //

      evc->input(rgbin);

      fps_timer.nextFrame();
      uidata.time_state = fps_timer.getState();

      if (uidata.time_state.frame_number % 50 == 0)
        LINFO("frame %u: %.2f fps", uidata.time_state.frame_number, uidata.time_state.recent_fps);

      const Image<byte> vcxmap = evc->getVCXmap();

      //
      // build a ui job to run in the background to display update the
      // saliency map the input frame, the vcx maps,
      //

      uiq->enqueueJob(rutz::make_shared
                      (new Nv2UiJob
                       (ofs.get(),
                        esv.get(),
                        eit.get(),
                        uidata,
                        esm.get(),
                        esec->getSeg().get(),
                        neoBrain.get(),
                        rgbin, vcxmap
                        , evc->getImap()
                        , evc->getCmap()
                        , evc->getOmap()
#ifdef ENV_WITH_DYNAMIC_CHANNELS
                        , evc->getFmap()
                        , evc->getMmap()
#endif
                        )));
    }

  // destroy the ui queue so that we force it to shut down now
  uiq.reset(0);

  manager.stop();

  return retval;
}

// ######################################################################
int main(int argc, const char** argv)
{
  try {
    return submain(argc, argv);
  } catch (...) {
    REPORT_CURRENT_EXCEPTION;
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
