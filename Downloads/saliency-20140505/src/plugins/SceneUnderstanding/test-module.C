/*! @file SceneUnderstanding/test-tensor.C Test the various vision comp
 * with simple stimulus*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/test-module.C $
// $Id: test-module.C 13413 2010-05-15 21:00:11Z itti $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "plugins/SceneUnderstanding/TensorVoting.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/SFS.H"
#include "Raster/Raster.H"
#include "GUI/DebugWin.H"
#include "GUI/ViewPort.H"
#include <GL/glut.h>

#include <signal.h>
#include <sys/types.h>

typedef unsigned char BYTE;

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);


typedef struct
{
  int type;
  unsigned int    maxX,
                  maxY;
  unsigned char   *image;
}       PIC;

void drawQube()
{
  glBegin(GL_QUADS);
  glColor4f(1, 1, 1, 1);

  // face v0-v1-v2-v3
  glNormal3f(0,0,1);
  glTexCoord2f(1, 1);  glVertex3f(1,1,1);
  glTexCoord2f(0, 1);  glVertex3f(-1,1,1);
  glTexCoord2f(0, 0);  glVertex3f(-1,-1,1);
  glTexCoord2f(1, 0);  glVertex3f(1,-1,1);

  // face v0-v3-v4-v5
  glNormal3f(1,0,0);
  glTexCoord2f(0, 1);  glVertex3f(1,1,1);
  glTexCoord2f(0, 0);  glVertex3f(1,-1,1);
  glTexCoord2f(1, 0);  glVertex3f(1,-1,-1);
  glTexCoord2f(1, 1);  glVertex3f(1,1,-1);

  // face v0-v5-v6-v1
  glNormal3f(0,1,0);
  glTexCoord2f(1, 0);  glVertex3f(1,1,1);
  glTexCoord2f(1, 1);  glVertex3f(1,1,-1);
  glTexCoord2f(0, 1);  glVertex3f(-1,1,-1);
  glTexCoord2f(0, 0);  glVertex3f(-1,1,1);

  // face  v1-v6-v7-v2
  glNormal3f(-1,0,0);
  glTexCoord2f(1, 1);  glVertex3f(-1,1,1);
  glTexCoord2f(0, 1);  glVertex3f(-1,1,-1);
  glTexCoord2f(0, 0);  glVertex3f(-1,-1,-1);
  glTexCoord2f(1, 0);  glVertex3f(-1,-1,1);

  // face v7-v4-v3-v2
  glNormal3f(0,-1,0);
  glTexCoord2f(0, 0);  glVertex3f(-1,-1,-1);
  glTexCoord2f(1, 0);  glVertex3f(1,-1,-1);
  glTexCoord2f(1, 1);  glVertex3f(1,-1,1);
  glTexCoord2f(0, 1);  glVertex3f(-1,-1,1);

  // face v4-v7-v6-v5
  glNormal3f(0,0,-1);
  glTexCoord2f(0, 0);  glVertex3f(1,-1,-1);
  glTexCoord2f(1, 0);  glVertex3f(-1,-1,-1);
  glTexCoord2f(1, 1);  glVertex3f(-1,1,-1);
  glTexCoord2f(0, 1);  glVertex3f(1,1,-1);
  glEnd();
}

void displayCB()
{

  int screenWidth = 320;
  int screenHeight = 240;

  LINFO("Draw");
  //Draw Qube
  glViewport(0, 0, screenWidth, screenHeight);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0f, (float)(screenWidth)/screenHeight, 1.0f, 100.0f);
  glMatrixMode(GL_MODELVIEW);

  // clear framebuffer
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // | GL_STENCIL_BUFFER_BIT);


  //Draw to back buffer
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_PIXEL_MODE_BIT); // for GL_DRAW_BUFFER and GL_READ_BUFFER
  glDrawBuffer(GL_BACK);
  glReadBuffer(GL_BACK);


  // object position and rot
  glPushMatrix();
  glTranslatef(0, 0, 1.5);
  glRotatef(45, 1, 0, 0);   // pitch
  glRotatef(45, 0, 1, 0);   // heading
  // draw a cube with the dynamic texture
  drawQube();
  glPopMatrix();



  glPopAttrib(); // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT

  // draw info messages
  //glutSwapBuffers();

  glFlush();


}

void setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, 0, 1, 0); // eye(x,y,z), focal(x,y,z), up(x,y,z)
}

void initLights()
{
    // set up light colors (ambient, diffuse, specular)
    GLfloat lightKa[] = {.2f, .2f, .2f, 1.0f};  // ambient light
    GLfloat lightKd[] = {.7f, .7f, .7f, 1.0f};  // diffuse light
    GLfloat lightKs[] = {1, 1, 1, 1};           // specular light
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

    // position the light
    float lightPos[4] = {0, 0, 20, 1}; // positional light
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
}


PIC UCFReadPic(const char* filename)
{
  FILE* infile = fopen(filename,"r");
  PIC temp;
  int ret;

  /* getting Type from image data */
  ret=fread(&temp.type,sizeof(temp.type),1,infile);
  switch (temp.type)
  {
    case 0xF10F:
    case 0xF200:
    case 0xF201:
    case 0xF204:
    case 0x0000:
      {
        //fread(&temp.maxX,sizeof(temp.maxX),1,infile);
        //fread(&temp.maxY,sizeof(temp.maxY),1,infile);
        unsigned char byte1,byte2,byte3,byte4;
        ret=fread(&byte1,sizeof(BYTE),1,infile);
        ret=fread(&byte2,sizeof(BYTE),1,infile);
        ret=fread(&byte3,sizeof(BYTE),1,infile);
        ret=fread(&byte4,sizeof(BYTE),1,infile);
        temp.maxX= byte1*16777216+byte2*65536+byte3*256+byte4;
        ret=fread(&byte1,sizeof(BYTE),1,infile);
        ret=fread(&byte2,sizeof(BYTE),1,infile);
        ret=fread(&byte3,sizeof(BYTE),1,infile);
        ret=fread(&byte4,sizeof(BYTE),1,infile);
        temp.maxY= byte1*16777216+byte2*65536+byte3*256+byte4;
        printf("Max: %i %i\n", temp.maxX, temp.maxY);
        break;
      }
    case 0x8000:
    case 0x8001:
    case 0xB003:
    default  :
      {
        ret=fread(&temp.maxX,sizeof(temp.maxX),1,infile);
        ret=fread(&temp.maxY,sizeof(temp.maxY),1,infile);
        break;
      }
  }
  if((temp.image=(BYTE*)calloc(temp.maxX*temp.maxY,sizeof(BYTE)))==NULL)
  {
    temp.maxX = temp.maxY = 0;
    temp.image = NULL;
    return(temp);
  }

  ret=fread(temp.image,sizeof(BYTE),temp.maxX * temp.maxY,infile);
  fclose(infile);
  return(temp);
}



int main(const int argc, const char **argv)
{

  //MYLOGVERB = LOG_INFO;
  //ModelManager manager("Test Vision");

  //nub::ref<SimEventQueueConfigurator>
  //  seqc(new SimEventQueueConfigurator(manager));
  //manager.addSubComponent(seqc);

  //nub::ref<SimOutputFrameSeries> ofs(new SimOutputFrameSeries(manager));
  //manager.addSubComponent(ofs);

  //nub::ref<SimInputFrameSeries> ifs(new SimInputFrameSeries(manager));
  //manager.addSubComponent(ifs);

  //nub::ref<TensorVoting> tensorVoting(new TensorVoting(manager));
  //nub::ref<V2> v2(new V2(manager));

 // nub::ref<SFS> sfs(new SFS(manager));

   // Request a bunch of option aliases (shortcuts to lists of options):
  //REQUEST_OPTIONALIAS_NEURO(manager);

  //if (manager.parseCommandLine(
  //      (const int)argc, (const char**)argv, "", 0, 0) == false)
  //  return 1;

  //nub::ref<SimEventQueue> seq = seqc->getQ();


  //Image<PixRGB<byte> > img = Raster::ReadRGB("/home/elazary/images/testImages/stream-output000000.pnm");
 // Image<PixRGB<byte> > img = Raster::ReadRGB("/home/elazary/images/testImages/frame.pnm");
 //

  //Read the ucf image
//  PIC pic1 = UCFReadPic("/home/elazary/shapeFromShading/shading/ucfimgs/sphere128.ucf");
//  Image<byte> img(pic1.maxX, pic1.maxY, NO_INIT);
//  for(uint j=0; j<pic1.maxY; j++)
//    for(uint i=0; i<pic1.maxX; i++)
//      img.setVal(i,j, pic1.image[i*pic1.maxX+j]);

  //manager.start();

//  sfs->evolve(img);
  //v2->evolve(tf);
  //v2->evolve(img);
  //tensorVoting->evolve();


  Display *display = XOpenDisplay(0);

  if ( !display )
  {
    printf( "Failed to open X display\n" );
    exit(1);
  }

  // Get a matching FB config
  static int visual_attribs[] =
    {
      GLX_X_RENDERABLE    , True,
      GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
      GLX_RENDER_TYPE     , GLX_RGBA_BIT,
      GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
      GLX_RED_SIZE        , 8,
      GLX_GREEN_SIZE      , 8,
      GLX_BLUE_SIZE       , 8,
      GLX_ALPHA_SIZE      , 8,
      GLX_DEPTH_SIZE      , 24,
      GLX_STENCIL_SIZE    , 8,
      GLX_DOUBLEBUFFER    , True,
      //GLX_SAMPLE_BUFFERS  , 1,
      //GLX_SAMPLES         , 4,
      None
    };

  printf( "Getting matching framebuffer configs\n" );
  int fbcount;
  GLXFBConfig *fbc = glXChooseFBConfig( display, DefaultScreen( display ),
                                        visual_attribs, &fbcount );
  if ( !fbc )
  {
    printf( "Failed to retrieve a framebuffer config\n" );
    exit(1);
  }
  printf( "Found %d matching FB configs.\n", fbcount );

  // Pick the FB config/visual with the most samples per pixel
  printf( "Getting XVisualInfos\n" );
  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

  int i;
  for ( i = 0; i < fbcount; i++ )
  {
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
    if ( vi )
    {
      int samp_buf, samples;
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );

      printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
              " SAMPLES = %d\n",
              i, (uint)vi -> visualid, samp_buf, samples );

      if ( (best_fbc < 0 || samp_buf ) && (samples > best_num_samp) )
        best_fbc = i, best_num_samp = samples;
      if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
        worst_fbc = i, worst_num_samp = samples;
    }
    XFree( vi );
  }

  // Get a visual
  int fbc_id = best_fbc;
  //int fbc_id = worst_fbc;

  XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[ fbc_id ]  );
  printf( "Chosen visual ID = 0x%x\n", (uint)vi->visualid );

  printf( "Creating colormap\n" );
  XSetWindowAttributes swa;
  swa.colormap = XCreateColormap( display, RootWindow( display, vi->screen ),
                                  vi->visual, AllocNone );
  swa.background_pixmap = None ;
  swa.border_pixel      = 0;
  swa.event_mask        = StructureNotifyMask;

  printf( "Creating window\n" );
  Window win = XCreateWindow( display, RootWindow( display, vi->screen ),
                              0, 0, 320, 240, 0, vi->depth, InputOutput,
                              vi->visual,
                              CWBorderPixel|CWColormap|CWEventMask, &swa );
  if ( !win )
  {
    printf( "Failed to create window.\n" );
    exit(1);
  }

  XStoreName( display, win, "GL 3.0 Window");

  printf( "Mapping window\n" );
  XMapWindow( display, win );

  // See if GL driver supports glXCreateContextAttribsARB()
  //   Create an old-style GLX context first, to get the correct function ptr.
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;

  GLXContext ctx_old = glXCreateContext( display, vi, 0, True );
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
           glXGetProcAddress( (const GLubyte *) "glXCreateContextAttribsARB" );

  GLXContext ctx = ctx_old;
  XFree( fbc );

  // Verifying that context is a direct context
  printf( "Verifying that context is direct\n" );
  if ( ! glXIsDirect ( display, ctx ) )
  {
    printf( "Indirect GLX rendering context obtained" );
    exit(1);
  }

  printf( "Making context current\n" );
  glXMakeCurrent( display, win, ctx );



  for(uint i=0; i<10; i++)
  {
    displayCB();
    glXSwapBuffers ( display, win );

    Image<PixRGB<byte> > img(320,240, ZEROS);

    //glPixelStorei(GL_PACK_ALIGNMENT,1);
    //glReadBuffer(GL_BACK);
    glReadPixels (0, 0, 320, 240,
        GL_RGB, GL_UNSIGNED_BYTE, (unsigned char*)img.getArrayPtr());
    SHOWIMG(img);
  }
  // stop all our ModelComponents
  //manager.stop();

  return 0;
}

