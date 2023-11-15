/*!@file saliency.C simple Maemo saliency app to run on the Nokia N810 */

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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>

#include <linux/videodev.h> // needed for some type definitions??
#include <linux/videodev2.h>

#include <linux/fb.h>
#include <asm-arm/arch-omap/omapfb.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define INT_IS_32_BITS 1
#define LONG_IS_32_BITS 1
#include "Envision/env_config.h"
#include "Envision/env_alloc.h"
#include "Envision/env_c_math_ops.h"
#include "Envision/env_stdio_interface.h"
#include "Envision/env_image.h"
#include "Envision/env_image_ops.h"
#include "Envision/env_log.h"
#include "Envision/env_visual_cortex.h"
#include "Envision/env_params.h"

#include <pthread.h>

#include "Envision/env_alloc.c"
#include "Envision/env_c_math_ops.c"
#include "Envision/env_stdio_interface.c"
#include "Envision/env_image.c"
#include "Envision/env_image_ops.c"
#include "Envision/env_log.c"
#include "Envision/env_visual_cortex.c"
#include "Envision/env_params.c"
#include "Envision/env_channel.c"
#include "Envision/env_math.c"
#include "Envision/env_motion_channel.c"
#include "Envision/env_pyr.c"

#include "Image/font6x10.h"
#define FONTW 6
#define FONTH 10

#include <libosso.h>

#define LINFO printf("\n")&&printf
#define LFATAL printf("\n")&&printf
#define PLFATAL printf("\n")&&printf
#define PLERROR printf("\n")&&printf

// grab: format is RGB565
#define GRABDEV "/dev/video0"
#define GRABNBUF 1
#define GRABW 320
#define GRABH 240

// display: format is RGB565
#define HRES 800
#define VRES 480
#define BPP 16

#define BCOL intg16(0x3FFF)
#define TCOL intg16(0x3F3F)
#define TCOL2 intg16(0xFF3F)
#define SCOL intg16(0x3F30)

#define OMAPFB_FORMAT_FLAG_TEARSYNC 0x0200
#define OMAPFB_FORMAT_FLAG_FORCE_VSYNC 0x0400

// Some of the code here is heavily inspired from this message:
// http://lists.maemo.org/pipermail/maemo-developers/2008-February/050357.html


// ######################################################################
// Thunk to convert from env_size_t to size_t
static void* malloc_thunk(env_size_t n)
{ return malloc(n); }

// ######################################################################
void draw_rect(intg16 *buf, const int x, const int y, const int w, const int h, const intg16 col)
{
  intg16 *b = buf + x + y * HRES;

  const int offy = (h-1) * HRES;
  for (int xx = 0; xx < w; ++xx) { b[xx] = col; b[xx + offy] = col; }

  const int offx = w-1;
  for (int yy = 0; yy < h * HRES; yy += HRES) { b[yy] = col; b[yy + offx] = col; }
}

// ######################################################################
void draw_filled_rect(intg16 *buf, const int x, const int y, const int w, const int h, const intg16 col)
{
  intg16 *b = buf + x + y * HRES;

  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) *b++ = col;
    b += HRES - w;
  }
}

// ######################################################################
void write_text(intg16 *buf, const char *txt, int x0, int y0, const intg16 col)
{
  const int len = int(strlen(txt));

  // auto centering?
  if (x0 == -1) x0 = (HRES - FONTW * len) / 2;

  for (int i = 0; i < len; i ++)
    {
      const unsigned char *ptr = ((const unsigned char *)font6x10) + (txt[i] - 32) * FONTW * FONTH;

      for (int y = 0; y < FONTH; y ++)
        for (int x = 0; x < FONTW; x ++)
          if (!ptr[y * FONTW + x]) buf[x0 + x + HRES * (y0 + y)] = col; else buf[x0 + x + HRES * (y0 + y)] = 0;
      x0 += FONTW;
    }
}

// ######################################################################
void draw_map(intg16 *buf, const env_image *img, const int xoff, const int yoff, const env_size_t scale)
{
  intg16 *d = buf + xoff + yoff * HRES;;
  intg32 *s = img->pixels;

  const env_size_t w = img->dims.w, h = img->dims.h;
  const env_size_t ws = w * scale;

  for (env_size_t jj = 0; jj < h; ++jj) {
    const intg16 *dd = d;
    for (env_size_t ii = 0; ii < w; ++ii) {
      const intg16 val = intg16( (*s++) >> 3 );
      for (env_size_t k = 0; k < scale; ++k) *d++ = val;
    }
    d += HRES - ws;
    for (env_size_t k = 1; k < scale; ++k) { memcpy(d, dd, ws * 2); d += HRES; }
  }
  draw_rect(buf, xoff, yoff, scale * w, scale * h, BCOL);
}

// ######################################################################
void print_help(intg16 *buf, const bool doit) {
  draw_filled_rect(buf, 0, 450, 800, 30, intg16(0)); // clear any old help message
  if (doit) {
    write_text(buf, "saliency  -  Copyright (c) 2009 by Laurent Itti and the iLab team  -  See http://iLab.usc.edu for more info about visual saliency", -1, 450, TCOL2);
    write_text(buf, "This program analyzes the input video to determine which point is most likely to attract human visual attention and gaze", -1, 460, TCOL2);
    write_text(buf, "Press <SPACE> for more info.              Press <ESC> or square key at center of cursor pad to exit application", -1, 470, TCOL2);
  }
}

// ######################################################################
void print_help2(intg16 *buf, const bool doit) {
  draw_filled_rect(buf, 0, 300, 800, 180, intg16(0)); // clear any old help message
  if (doit) {
    //---------------0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
    write_text(buf, "saliency  -  Copyright (c) 2009 by Laurent Itti and the iLab team  -  See http://iLab.usc.edu for more info about visual saliency", -1, 310, TCOL2);
    write_text(buf, "This program analyzes the input video to determine which point is most likely to attract human visual attention and gaze.", -1, 320, TCOL2);
    //write_text(buf, "", -1, 330, TCOL2);
    write_text(buf, "In this biologically-inspired system (Itti, Koch & Niebur, IEEE Transactions on Pattern Analysis and Machine Intelligence, 1998),", -1, 340, TCOL2);
    write_text(buf, "each input image is decomposed into a set of multiscale neural ``feature maps,'' which extract local spatial discontinuities in the", -1, 350, TCOL2);
    write_text(buf, "modalities of color, intensity, orientation, flicker and motion. Each feature map is endowed with non-linear spatially competitive", -1, 360, TCOL2);
    write_text(buf, "dynamics, so that the response of a neuron at a given location in a map is modulated by the activity of neighboring neurons. Such", -1, 370, TCOL2);
    write_text(buf, "contextual modulation, inspired by recent neurobiological findings, enhances salient targets from cluttered backgrounds. All feature", -1, 380, TCOL2);
    write_text(buf, "maps are then combined into a unique scalar saliency map which encodes for the salience of a location in the scene, irrespectively", -1, 390, TCOL2);
    write_text(buf, "of the particular feature which detected this location as conspicuous. A winner-take-all neural network then detects the point of", -1, 400, TCOL2);
    write_text(buf, "highest salience in the map at any given time, and draws the focus of attention towards this location (small cyan square marker).", -1, 410, TCOL2);
    write_text(buf, "", -1, 420, TCOL2);
    write_text(buf, "This Maemo program uses a fast integer-only saliency algorithm implemented by Robert J. Peters, see Peters & Itti, ACM Transactions", -1, 430, TCOL2);
    write_text(buf, "on Applied Perception, 2008.", -1, 440, TCOL2);
    //write_text(buf, "", -1, 450, TCOL2);
    //write_text(buf, "", -1, 460, TCOL2);
    write_text(buf, "Press <SPACE> to switch back to full display mode.", -1, 470, TCOL2);
  }
}

/**************************************************************************************************************/
int main(int argc, char **argv)
{
  // initialize OSSO, we will need it to prevent the screen from dimming:
  osso_context_t *osso = osso_initialize("saliency", "1.00", true, 0);
  if (osso == NULL) LFATAL("Cannot initialize OSSO");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // setup V4L2 mmap'ed video grabbing:
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int gfd = open(GRABDEV, O_RDWR);
  if (gfd == -1) PLFATAL("Cannot open V4L2 device %s", GRABDEV);

  // set grab format:
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = GRABW;
  fmt.fmt.pix.height = GRABH;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;

  if (ioctl(gfd, VIDIOC_S_FMT, &fmt) == -1) PLFATAL("Cannot set requested video mode/resolution");

  // prepare mmap interface:
  const int nbuf = GRABNBUF;
  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = nbuf;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (ioctl(gfd, VIDIOC_REQBUFS, &req) == -1) PLFATAL("Cannot allocate %d mmap'ed video frame buffers", nbuf);
  if (int(req.count) != nbuf) LFATAL("Hardware only supports %d video buffers (vs. %d requested)", req.count, nbuf);

  byte **itsMmapBuf = new byte*[req.count];
  int *itsMmapBufSize = new int[req.count];

  for (uint i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (ioctl(gfd, VIDIOC_QUERYBUF, &buf) == -1) PLFATAL("Could not query for MMAP buffer");

    itsMmapBufSize[i] = buf.length;
    itsMmapBuf[i] = static_cast<byte*>(mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, gfd, buf.m.offset));
    if (itsMmapBuf[i] == MAP_FAILED) PLFATAL("Error MMAP'ing video buffer number %d", i);
  }

  // get ready to grab frames, starting with buffer/frame 0:
  int itsCurrentFrame = 0;
  bool *itsGrabbing = new bool[nbuf];
  for (int i = 0; i < nbuf; ++i) itsGrabbing[i] = false;

  // start streaming by putting grab requests for all our buffers:
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  for (int i = 0; i < nbuf; ++i)
    if (itsGrabbing[i] == true) {
      buf.index = i;
      if (ioctl(gfd, VIDIOC_DQBUF, &buf) == -1) PLFATAL("VIDIOC_DQBUF (frame %d)", i);
      itsGrabbing[i] = false;
    }

  for (int i = 0; i < nbuf; ++i) {
    // now start a fresh grab for buffer i:
    buf.index = i;
    if (ioctl(gfd, VIDIOC_QBUF, &buf)) PLFATAL("VIDIOC_QBUF (frame %d)", i);
    itsGrabbing[i] = true;
  }

  // tell grabber to stream:
  enum v4l2_buf_type typ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(gfd, VIDIOC_STREAMON, &typ)) PLFATAL("VIDIOC_STREAMON");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // setup display via direct OMAPFB access:
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // setup display, screen and window
  Display *display = XOpenDisplay(getenv ("DISPLAY"));
  if (display == NULL) LFATAL("cannot open X display");
  int screen_num = DefaultScreen(display);

  Window win = XCreateSimpleWindow(display, RootWindow(display, screen_num), 0, 0, HRES, VRES, 0,
                                   WhitePixel(display, screen_num), BlackPixel (display, screen_num));
  XMapWindow(display, win);
  XSelectInput(display, win, ExposureMask | KeyPressMask);
  XFlush(display);

  XEvent xev;
  XWindowEvent(display, win, ExposureMask, &xev);

  // bring window to fullscreen
  xev.xclient.type = ClientMessage;
  xev.xclient.serial = 0;
  xev.xclient.send_event = True;
  xev.xclient.message_type = XInternAtom (display, "_NET_WM_STATE", False);
  xev.xclient.window = win;
  xev.xclient.format = 32;
  xev.xclient.data.l[0] = 1;
  xev.xclient.data.l[1] = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);
  xev.xclient.data.l[2] = 0;
  xev.xclient.data.l[3] = 0;
  xev.xclient.data.l[4] = 0;

  if (!XSendEvent(display, DefaultRootWindow(display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev))
    LFATAL("cannot bring X window to fullscreen");
  XSync(display, False);

  // open framebuffer device
  int fbfd = open("/dev/fb0", O_RDWR);
  if (!fbfd) LFATAL("cannot open framebuffer device");

  size_t ssize = HRES * BPP / 8 * VRES;

  // map framebuffer
  char* fbp = (char*)mmap(0, ssize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
  if ((int)fbp == -1) LFATAL("failed to memory map framebuffer");
  intg16 *fbp16 = (intg16 *)fbp;

  // setup fullscreen update info struct
  struct omapfb_update_window update;

  // copy full screen from fb to lcd video ram
  update.x = 0;
  update.y = 0;
  update.width = HRES;
  update.height = VRES;

  // request native pixel format, tearsync and vsync
  update.format = OMAPFB_COLOR_RGB565 | OMAPFB_FORMAT_FLAG_TEARSYNC; // | OMAPFB_FORMAT_FLAG_FORCE_VSYNC;


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Setup Envision:
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Instantiate our various ModelComponents:
  struct env_params envp;
  env_params_set_defaults(&envp);

  envp.maxnorm_type = ENV_VCXNORM_MAXNORM;
  envp.scale_bits = 16;
  env_allocation_init(&malloc_thunk, &free);

  struct env_visual_cortex ivc;
  env_visual_cortex_init(&ivc, &envp);
  struct env_dims indims; indims.w = GRABW; indims.h = GRABH;
  struct env_rgb_pixel* input = (struct env_rgb_pixel*)env_allocate(GRABW * GRABH * sizeof(struct env_rgb_pixel));

  // initially print help message at bottom:
  int helpmode = 30; print_help(fbp16, true);
  bool helpmode2 = false; bool refresh = true;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Main loop:
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  for (int fram = 0; /* */ ; ++fram)
    {
      //////////////////// grab
      intg16* result = (intg16*)(itsMmapBuf[itsCurrentFrame]);
      // int siz = itsMmapBufSize[itsCurrentFrame];
      struct v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = itsCurrentFrame;

      // are we already grabbing buffer 'itsCurrentFrame'? otherwise,
      // start the grab now:
      if (itsGrabbing[itsCurrentFrame] == false) {
        if (ioctl(gfd, VIDIOC_QBUF, &buf)) PLFATAL("VIDIOC_QBUF (frame %d)", itsCurrentFrame);
        itsGrabbing[itsCurrentFrame] = true;
      }

      // wait until buffer 'itsCurrentFrame' has been fully captured:
      if (ioctl(gfd, VIDIOC_DQBUF, &buf) == -1) PLFATAL("VIDIOC_DQBUF (frame %d)", itsCurrentFrame);
      itsGrabbing[itsCurrentFrame] = false;

      // get ready for capture of that frame again (for later):
      itsGrabbing[itsCurrentFrame] = true;
      if (ioctl(gfd, VIDIOC_QBUF, &buf) < 0) PLFATAL("VIDIOC_QBUF (frame %d)", itsCurrentFrame);

      // switch to another frame:
      ++itsCurrentFrame;
      if (itsCurrentFrame >= nbuf) itsCurrentFrame = 0;

      ///////////// saliency

      // convert image to RGB, rescaling the 5 or 6-bit values to full 8-bit:
      unsigned char *in = ((unsigned char *)input) + 2; // point to blue of first pixel
      intg16 *data = result;
      for (uint ii = 0; ii < GRABW * GRABH; ++ii) {
          intg16 x = *data++;
          *in-- = ((byte(x)) << 3) & byte(0xF8);   // blue
          x >>= 3; *in-- = (byte(x)) & byte(0xFC); // green
          x >>= 5; *in = (byte(x)) & byte(0xF8);   // red
          in += 5; // go to blue of next pixel
        }

      struct timeval real1, real2;
      gettimeofday(&real1, /* timezone */ 0);

      // process saliency map:
      struct env_image ivcout = env_img_initializer;
      struct env_image intens = env_img_initializer;
      struct env_image color = env_img_initializer;
      struct env_image ori = env_img_initializer;
#ifdef ENV_WITH_DYNAMIC_CHANNELS
      struct env_image flicker = env_img_initializer;
      struct env_image motion = env_img_initializer;
#endif

      env_visual_cortex_input(&ivc, &envp, "visualcortex", input, 0, indims, 0, 0, &ivcout, &intens, &color, &ori
#ifdef ENV_WITH_DYNAMIC_CHANNELS
                              , &flicker, &motion
#endif
                              );

      env_visual_cortex_rescale_ranges(&ivcout, &intens, &color, &ori
#ifdef ENV_WITH_DYNAMIC_CHANNELS
                                       , &flicker, &motion
#endif
);

      gettimeofday(&real2, /* timezone */ 0);
      const double real_secs = (real2.tv_sec - real1.tv_sec) + (real2.tv_usec - real1.tv_usec) / 1000000.0;

      if (helpmode2 == false) {
        if (helpmode == 0) {
          char msg[255];
          sprintf(msg, "frame %06d, %0.2ffps", fram, 1.0 / real_secs);
          write_text(fbp16, msg, -1 /*336*/, 464, TCOL);
        } else { if (--helpmode == 0) print_help(fbp16, false); } // clear help message
      }

      //////////////////// render

      // "render" whole frame, we scale it up by 25% to 400x300 by duplicating every 4th row & column:
      intg16 *bf = fbp16;
      intg16 *img = result;
      for (uint jj = 0; jj < GRABH/4; ++jj) {
        // copy first row twice:
        for (uint ii = 0; ii < GRABW/4; ++ii)
          { *bf++ = *img; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; } bf+=HRES-400; img-=GRABW;
        for (uint ii = 0; ii < GRABW/4; ++ii)
          { *bf++ = *img; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; } bf += HRES - 400;
        // then the next 3 once:
        for (uint ii = 0; ii < GRABW/4; ++ii)
          { *bf++ = *img; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; } bf += HRES - 400;
        for (uint ii = 0; ii < GRABW/4; ++ii)
          { *bf++ = *img; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; } bf += HRES - 400;
        for (uint ii = 0; ii < GRABW/4; ++ii)
          { *bf++ = *img; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; *bf++ = *img++; } bf += HRES - 400;
      }

      draw_rect(fbp16, 0, 0, 400, 300, BCOL);

      // draw the maps:
      draw_map(fbp16, &ivcout, 400, 0, 20);
      write_text(fbp16, "saliency map", 800 - 12*6-3, 3, TCOL);
      if (helpmode2 == false) {
        draw_map(fbp16, &intens, 0, 325, 8);
        draw_map(fbp16, &color, GRABW/2, 325, 8);
        draw_map(fbp16, &ori, GRABW, 325, 8);
        draw_map(fbp16, &flicker, (GRABW*3)/2, 325, 8);
        draw_map(fbp16, &motion, GRABW*2, 325, 8);
      }

      // find most salient point:
      intg32 *sm = ivcout.pixels;
      env_size_t mx = 0, my = 0;
      intg32 mv = *sm;
      const env_size_t smw = ivcout.dims.w, smh = ivcout.dims.h;
      for (env_size_t j = 0; j < smh; ++j)
        for (env_size_t i = 0; i < smw; ++i)
          if (*sm > mv) { mv = *sm++; mx = i; my = j; } else ++sm;
      draw_filled_rect(fbp16, mx * 20+6, my * 20 + 6, 8, 8, SCOL);
      draw_rect(fbp16, mx * 20+5, my * 20 + 5, 10, 10, 0);
      draw_filled_rect(fbp16, mx * 20+6 + 400, my * 20 + 6, 8, 8, SCOL);


      // display some text labels (only necessary on first frame since we won't erase them):
      if (refresh) {
        write_text(fbp16, "intensity", 80 - 30, 314, TCOL);
        write_text(fbp16, "color", 240 - 15, 314, TCOL);
        write_text(fbp16, "orientation", 400 - 33, 314, TCOL);
        write_text(fbp16, "flicker", 560 - 18, 314, TCOL);
        write_text(fbp16, "motion", 720 - 15, 314, TCOL);
      }

      env_img_make_empty(&ivcout);
      env_img_make_empty(&intens);
      env_img_make_empty(&color);
      env_img_make_empty(&ori);
      env_img_make_empty(&flicker);
      env_img_make_empty(&motion);

      // check for keypress, abort when the square buttonat middle of cursor is pressed:
      XEvent event; refresh = false;
      if (XCheckWindowEvent(display, win, KeyPressMask, &event) == True && event.type == KeyPress) {
        if (event.xkey.keycode == 0x68 || event.xkey.keycode == 0x9) break;  // square or esc
        else if (event.xkey.keycode == 0x41) { helpmode2 = !helpmode2; print_help2(fbp16, helpmode2); helpmode = 0; if (helpmode2 == false) refresh = true; } // space
        else if (helpmode2 == false) { helpmode = 20; print_help(fbp16, true); }
      }

      // wait for fb->lcd video ram transfer complete
      ioctl(fbfd, OMAPFB_SYNC_GFX);

      // wait for vsync
      ioctl (fbfd, OMAPFB_VSYNC);

      // request transfer of fb-> lcd video ram for whole screen
      ioctl (fbfd, OMAPFB_UPDATE_WINDOW, &update);

      // prevent display from automatically dimming:
      if (fram % 30 == 0) osso_display_blanking_pause(osso);
    }

  // cleanup
  env_deallocate(input);
  env_visual_cortex_destroy(&ivc);
  env_allocation_cleanup();

  munmap(fbp, ssize); close(fbfd);

  typ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(gfd, VIDIOC_STREAMOFF, &typ)) PLERROR("VIDIOC_STREAMOFF");

  for (int i = 0; i < nbuf; i ++) munmap(itsMmapBuf[i], itsMmapBufSize[i]);
  delete [] itsMmapBuf; itsMmapBuf = NULL;
  delete [] itsMmapBufSize; itsMmapBufSize = NULL;
  close(gfd);
  delete [] itsGrabbing;

  // cleanup X stuff
  XCloseDisplay (display);

  osso_deinitialize(osso);

  return 0;
}
