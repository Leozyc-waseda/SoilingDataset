/*!@file CUDA/CudaImageDisplayGL.C  Displays Images from CUDA Memory using OpenGL*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImageDisplayGL.C $
// $Id: CudaImageDisplayGL.C 14165 2010-10-23 07:00:08Z rand $
//


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include "CUDA/CudaImageDisplayGL.H"
#include "CUDA/CudaImage.H"
#include "CUDA/CudaImageSet.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"

#include "CudaImageDisplay.h"
#include <unistd.h>

#include "CUDA/CudaCutPaste.H"
#include "CUDA/CudaSaliency.H"
#include "CUDA/CudaMathOps.H"



#define BUFFER_DATA(i) ((char *)0 + i)

static const char *shader_code =
  "!!ARBfp1.0\n"
  "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
  "END";

CudaImageDisplayGL *CudaImageDisplayGL::instance = NULL;

CudaImageDisplayGL::CudaImageDisplayGL()
{
  bufferIndex=0;
  count_buffer_runs=0;
  g_Kernel = 0;
  g_FPS = false;
  g_Diag = false;
  frameN = 24;
  frameCounter = 0;
  shutdown = false;
}

// ######################################################################
void CudaImageDisplayGL::runImageFilters(unsigned int *d_dst)
{
  cuda_Copy(d_dst, mWinW, mWinH);
  CUT_CHECK_ERROR("Filtering kernel execution failed.\n");
}
// ######################################################################
void CudaImageDisplayGL::displayFunction()
{
  if(getShutdown())
    return;
  unsigned int *d_dst = NULL;

  CUDA_SAFE_CALL( cudaGLMapBufferObject((void**)&d_dst, gl_PBO ) );
  CUDA_SAFE_CALL( CUDA_Bind2TextureArray(bufferIndex));
  runImageFilters(d_dst);
  CUDA_SAFE_CALL(CUDA_UnbindTexture(bufferIndex));
  CUDA_SAFE_CALL(cudaGLUnmapBufferObject(gl_PBO));


  // Common display code path
  {
    glClear(GL_COLOR_BUFFER_BIT);
    //glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, mWinW, mWinH, GL_RGBA, GL_UNSIGNED_BYTE,BUFFER_DATA(0) );

    glBegin(GL_QUADS);
    glTexCoord2f(0,1); glVertex2f(-1, +1);
    glTexCoord2f(1,1); glVertex2f(1, +1);
    glTexCoord2f(1, 0); glVertex2f(+1, -1);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glEnd();

    glFinish();
  }

  if(frameCounter == frameN){
    frameCounter = 0;
    if(g_FPS){

      g_FPS = false;
    }
  }

  glutSwapBuffers();
  glutPostRedisplay();

}


// ######################################################################
void CudaImageDisplayGL::shutDown()
{
  CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(gl_PBO) );
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glDeleteBuffers(1, &(gl_PBO));
  glDeleteTextures(1, &(gl_Tex));
  free(h_Src);
  CUDA_SAFE_CALL( CUDA_FreeArray() );
  framework.setMutexDestroy();
  printf("Shutdown done.\n");
  cudaThreadExit();
}

// ######################################################################
int CudaImageDisplayGL::initGL(int x_size,int y_size)
{

  printf("Initializing GLUT...\n");
  int tmpImageW = x_size;
  int tmpImageH = y_size;
  int argc = 1;
  char **argv = new char*[2];
  argv[0] = new char[50];
  sprintf(argv[0],"INVALID COMMAND LINE OPTIONS");
  argv[1]=NULL;
  glutInit(&argc, argv);
  delete argv[0];
  delete[] argv;
  printf("%s %d\n",__FILE__,__LINE__);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);printf("%s %d\n",__FILE__,__LINE__);
  glutInitWindowSize(tmpImageW,tmpImageH);printf("%s %d\n",__FILE__,__LINE__);
  glutInitWindowPosition(-1,-1); printf("%s %d\n",__FILE__,__LINE__);

  main_window = glutCreateWindow("Main Display");
  glutDisplayFunc(displayWrapper);


  printf("OpenGL window created.\n");

  glewInit();
  printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

    if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
      fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
      fprintf(stderr, "This sample requires:\n");
      fprintf(stderr, "  OpenGL version 1.5\n");
      fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
      fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
      fflush(stderr);
      return -1;
  }

  return 0;
}

// ######################################################################
//Compile the Assembly share code
GLuint CudaImageDisplayGL::compileASMShader(GLenum program_type, const char *code)
{
  GLuint program_id;
  glGenProgramsARB(1, &program_id);
  glBindProgramARB(program_type, program_id);
  glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

  GLint error_pos;
  glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
  if (error_pos != -1) {
    const GLubyte *error_string;
    error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
    fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
    return 0;
  }
  return program_id;
}


// ######################################################################
void CudaImageDisplayGL::initAllTex()
{
  glEnable(GL_TEXTURE_2D);
  glGenTextures(MAX_SIZE, &gl_Tex);
  glGenBuffers(MAX_SIZE,&gl_PBO);

}


// ######################################################################
void CudaImageDisplayGL::initOpenGLBuffers()
{
  glEnable(GL_TEXTURE_2D);
  printf("Creating GL texture...\n");
  //Generating Texture
  glGenTextures(1, &(gl_Tex));
  //Binding Texture to GL_TEXTURE_2D
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  //Setting up paramters for the texture
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //Allocating data from the Texture to h_Src {Texture->h_Src}
  if(count_buffer_runs == 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,mWinW,mWinH,0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);

  printf("Creating PBO...\n");
  //Generating Pixel Buffer Object
  glGenBuffers(1, &(gl_PBO));
  //Binding Pixel Buffer Object to Unpack buffer
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  //Buffering Unpacked Buffer with h_Src data
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, mWinW * mWinH * 4, h_Src, GL_STREAM_COPY);
  //Register the PBO with CUDA
  CUDA_SAFE_CALL( cudaGLRegisterBufferObject(gl_PBO) );
  //CUT_CHECK_ERROR_GL();
  printf("PBO created.\n");
  //2 rotations for proper display
  glRotatef(180, 0, 0, 1);
  glRotatef(180, 0 ,1, 0);
  //load shader program
  shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB,shader_code);
  count_buffer_runs++;

}

// ######################################################################
void CudaImageDisplayGL::cleanup()
{
  glDeleteProgramsARB(1, &shader);
}


// ######################################################################
void CudaImageDisplayGL::initMainWindow()
{
  initGL(mWinW,mWinH);
}


// ######################################################################
void CudaImageDisplayGL::initDispGL(CudaImage<PixRGB<float> > &src)
{
  const int dev = src.getMemoryDevice();
  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  CudaImage<unsigned int> dst = CudaImage<unsigned int>(src.getDims(),NO_INIT, src.getMemoryPolicy(), dev);

  CUDA_convert_float_uint((float3_t *) src.getCudaArrayPtr(),(unsigned int *) dst.getCudaArrayPtr(),tile.sz(),src.size());
  CUDA_MallocArray((unsigned int *)dst.getCudaArrayPtr(),src.getWidth(),src.getHeight(),0);
  CUDA_MallocArray((unsigned int *)dst.getCudaArrayPtr(),src.getWidth(),src.getHeight(),1);
  initOpenGLBuffers();
}

void CudaImageDisplayGL::idleFunction()
{
  // Does nothing, subclasses might want to override
}

// ######################################################################
void CudaImageDisplayGL::updateDispGL()
{

  // Get the buffer that is not currently being used
  int newBufferIndex = (bufferIndex+1)%2;
  int w = framework.getCanvasW();
  int h = framework.getCanvasH();
  const Dims tile = CudaDevices::getDeviceTileSize1D(framework.getDev());
  CudaImage<unsigned int> dst = CudaImage<unsigned int>(Dims(w,h),NO_INIT, framework.getMP(), framework.getDev());
  // Critical section

  int lockAtt = framework.getLockAtt();
  switch(lockAtt)
    {
    case EINVAL:
    case EAGAIN:
    case EDEADLK:
      fprintf(stderr,"Canvas Lock is BROKEN!\n");
      exit(0);
      break;
    case EBUSY:
      // Ok, no updating this time
      break;
    default:
      // Copy new frame to the unused buffer
      //Get image handle
      CUDA_convert_float_uint((float3_t *)(framework.getCanvas()).getCudaArrayPtr(),(unsigned int *) dst.getCudaArrayPtr(),tile.sz(),w*h);
      framework.setCanvasModified(false);
      framework.setMutexUnlock();
      CUDA_UpdateArray((unsigned int *)dst.getCudaArrayPtr(),w,h,newBufferIndex) ;
      // Update buffer index so callback will now use this new buffer
      bufferIndex = newBufferIndex;
      glutPostRedisplay();
    }
}


bool CudaImageDisplayGL::getShutdown()
{
    return shutdown;
}

void CudaImageDisplayGL::setShutdown(bool isShutdown)
{
    shutdown = isShutdown;
}

// ######################################################################
void CudaImageDisplayGL::timerFunction(int index)
{
  if(getShutdown())
    return;
  if(framework.getCanvasModified())
    {
      //printf("Updating modified canvas\n");
      updateDispGL();
    }
  glutTimerFunc(1,timerWrapper,0);
}
// #####################################################################
float CudaImageDisplayGL::getFPS()
{
  return float(frameCounter)/float(tim.getSecs());
}

// ######################################################################
void CudaImageDisplayGL::createDisplay(int w, int h)
{
  mWinW = w;
  mWinH = h;

  initMainWindow();
  initDispGL(framework.getCanvas());
  glutTimerFunc(1,timerWrapper,0);
  tim.reset();
}




