#ifndef _NEOVISION2CUDA_H_DEFINED_
#define _NEOVISION2CUDA_H_DEFINED_

#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"
#include "Image/Point2D.H"
#include "NeovisionII/Nv2LabelReader.H"
#include "GUI/PrefsWindow.H"
#include "rutz/shared_ptr.h"
#include <unistd.h>
#include <vector>

class neovision2cuda : public CudaImageDisplayGL
{
public:
  static inline neovision2cuda *createCudaDisplay(nub::ref<InputFrameSeries> ifs_in, nub::ref<OutputFrameSeries> ofs_in, nub::ref<CudaSaliency> csm_in, std::vector<rutz::shared_ptr<Nv2LabelReader> > readers_in)
  {
    instance = (CudaImageDisplayGL*)new neovision2cuda(ifs_in,ofs_in,csm_in,readers_in);
    return (neovision2cuda*)instance;
  }
  void idleFunction();
  void runDisplay(int w, int h, MemoryPolicy mp, int dev);
protected:
  static inline void idleWrapper() { instance->idleFunction(); }
  static inline void displayWrapper() { instance->displayFunction(); }
  static inline void timerWrapper(int index) { instance->timerFunction(index); }
private:
  neovision2cuda(nub::ref<InputFrameSeries> ifs_in, nub::ref<OutputFrameSeries> ofs_in, nub::ref<CudaSaliency> csm_in, std::vector<rutz::shared_ptr<Nv2LabelReader> > readers_in);
  nub::ref<InputFrameSeries> ifs;
  nub::ref<OutputFrameSeries> ofs;
  nub::ref<CudaSaliency> csm;
  std::vector<rutz::shared_ptr<Nv2LabelReader> > readers;
  std::deque<Timer> frameTimes;
  PrefsWindow *pwin;
  //Sets points
  Point2D<int> rawFramePoint;
  Point2D<int> saliencyFramePoint;
  Point2D<int> imapFramePoint;
  Point2D<int> cmapFramePoint;
  Point2D<int> omapFramePoint;
  Point2D<int> fmapFramePoint;
  Point2D<int> mmapFramePoint;
  Point2D<int> inhibitionMapFramePoint;
  //Values regarding various maps in the GUI
  int n_cm; //Number of conspicuity maps
  int w_cm; //Width of conspicuity maps
  int h_cm; //Height of conspicuity maps
  int w_sm; //width of secondary maps (saliency and inhibition)
  int h_sm; //Height of secondary maps (salieny and inhibition)
  int n_sm; //Number of secondary maps
  int w_in; //Width of scaled input
  int h_in; //Height of scaled input
  double aspect_ratio; 
   
};

#endif
