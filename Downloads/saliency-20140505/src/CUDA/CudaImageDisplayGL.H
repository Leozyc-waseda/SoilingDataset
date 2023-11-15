/*!@file CUDA/CudaImageDisplayGL.H C++ wrapper for CUDA Image DisplayGL operations */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImageDisplayGL.H $
// $Id: CudaImageDisplayGL.H 13222 2010-04-15 00:39:05Z itti $
//

#ifndef CUDAIMAGEDISPLAYGL_H_DEFINED
#define CUDAIMAGEDISPLAYGL_H_DEFINED
#include "CUDA/cutil.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaFramework.H"
#include "CUDA/CudaImage.H"
#include "CUDA/wrap_c_cuda.h"
#include "CUDA/CudaColorOps.H"
#include "CUDA/CudaDevices.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaImageSetOps.H"
#include "Util/Timer.H"

class CudaImageDisplayGL
{
protected:
  static CudaImageDisplayGL *instance;

public:
  static inline CudaImageDisplayGL *createCudaDisplay()
  {
    instance = new CudaImageDisplayGL();
    return instance;
  }

  void createDisplay(int w, int h);
  void runImageFilters(unsigned int *d_dst);
  void registerIdleFunction();
  void shutDown();
  int initGL(int x_size,int y_size);
  GLuint compileASMShader(GLenum program_type, const char *code);
  void initAllTex();
  void initOpenGLBuffers();
  void cleanup();
  void initMainWindow();
  void initDispGL(CudaImage<PixRGB<float> > &src);
  void updateDispGL();
  // Graphics pipeline processing is here
  virtual void displayFunction();
  // Do non graphics processing here
  virtual void idleFunction();
  // Update the frame if the canvas has changed
  virtual void timerFunction(int index);
  float getFPS();
  bool getShutdown();
  void setShutdown(bool isShutdown);

protected:
  CudaFramework framework;
  int frameCounter;
  static inline void idleWrapper() { instance->idleFunction(); }
  static inline void displayWrapper() { instance->displayFunction(); }
  static void timerWrapper(int index) { instance->timerFunction(index); }
  // No instantiations, except from within subclass constructors
  CudaImageDisplayGL();

private:
  //OpenGL PBO and texture "names"
  GLuint gl_PBO,gl_Tex;
  Timer tim;
  //Source image on the host side
  unsigned int *h_Src;
  int mWinW,mWinH;
  GLuint shader;
  int bufferIndex;
  int main_window;
  MemoryPolicy mp;
  int count_buffer_runs;
  int  g_Kernel;
  bool g_FPS;
  bool g_Diag;
  int frameN;
  bool shutdown;
// ######################################################################
//shader for displaying floating-point texture
};
#endif
