/*! @file BeoSub/test-BeoSubDecoder.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubDecoder.C $
// $Id: test-BeoSubDecoder.C 14295 2010-12-02 20:02:32Z itti $

// compile with: -lm -lglut -lGLU -lGL -lX11 -lXext


#include <cmath>
#ifdef HAVE_GL_GLUT_H
#include <GL/gl.h>
#undef APIENTRY // otherwise it gets redefined between gl.h and glut.h???
#include <GL/glut.h>
#include <GL/glu.h>
#endif
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Component/ModelManager.H"
#include "BeoSub/BeoSubTaskDecoder.H"

#ifdef HAVE_GL_GLUT_H

float fangle=0.0,deltaAngle = 0.0,ratio, dtemp;
float x=0.0f,y=5.0f,z=2.0f;     // init postion of the camera
float lx=0.0f,ly=0.0f,lz=-0.1f;
bool shape = true;
int sw=320,sh=240;
int deltaMove = 0,h=300,w=300;
int deltaDepth = 0;
int counter=0;
void* font=GLUT_BITMAP_8_BY_13;
int bitmapHeight=13;
int frame,atime,timebase=0;
int outframe,outtime,outtimebase=0, decode = 0;
char s[30],s2[30];
int mainWindow, subWindow4;

float avgFR =0.0F;
byte *image = new byte[sw*sh*3];

ImageSet< PixRGB<byte> > inStream;

ModelManager mgr("BeoSubTaskDecoder Tester");
// Instantiate our various ModelComponents:
nub::soft_ref<BeoSubTaskDecoder> test(new BeoSubTaskDecoder(mgr));

void initWindow();

void changeSize2(int w1, int h1)
{
  // Prevent a divide by zero, when window is too short
  // (you cant make a window of zero width).
  ratio = 1.0f * w1 / h1;
  // Reset the coordinate system before modifying
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Set the viewport to be the entire window
  glViewport(0, 0, w1, h1);

  // Set the clipping volume
  gluPerspective(45,ratio,0.1,1000);
  glMatrixMode(GL_MODELVIEW);
}

void changeSize(int w1,int h1) {
  if(h1 == 0)
    h1 = 1;

  w = w1;
  h = h1;

  glutSetWindow(subWindow4);
  glutPositionWindow(0,0);
  glutReshapeWindow(w, h);
  changeSize2(w, h);

}

// rotate alone y axis
void drawRecRotate(GLfloat de, GLfloat x, GLfloat y, GLfloat z, GLfloat w, GLfloat d, GLfloat h)
{
  // rotation matrix
  // cos(de)  0  -sin(de)
  // 0          1  0
  // sin(de)  0  cos(de)
  // newx = cos(de) * x - sin(de) * z
  // newz = sin(de) * x + cos(de) * z
  // down
  GLfloat tempx=x,tempz=z;
  x=z=0;
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y, sin(de)*(x+w)+cos(de)*z+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y,  sin(de)*x+cos(de)*(z+d)+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y,  sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y, sin(de)*(x+w)+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glEnd();
  // up
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y+h, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y+h, sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y+h, sin(de)*(x+w)+cos(de)*z+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y+h, sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y+h, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y+h, sin(de)*(x+w)+cos(de)*z+tempz);
  glEnd();
  // left
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y, sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y+h, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y, sin(de)*x+cos(de)*z+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y+h, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y,  sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y+h, sin(de)*x+cos(de)*(z+d)+tempz);
  glEnd();
  // right
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y, sin(de)*(x+w)+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y+h, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y+h, sin(de)*(x+w)+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y+h, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y,  sin(de)*(x+w)+cos(de)*z+tempz);
  glEnd();
  // front
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y+h,  sin(de)*x+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y, sin(de)*x+cos(de)*(z+d)+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*(z+d)+tempx, y+h, sin(de)*(x+w)+cos(de)*(z+d)+tempz);
  glVertex3f( cos(de)*x-sin(de)*(z+d)+tempx, y+h,  sin(de)*x+cos(de)*(z+d)+tempz);
  glEnd();
  // back
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y+h,  sin(de)*(x+w)+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y, sin(de)*(x+w)+cos(de)*z+tempz);
  glEnd();
  glBegin(GL_TRIANGLES);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*x-sin(de)*z+tempx, y+h, sin(de)*x+cos(de)*z+tempz);
  glVertex3f( cos(de)*(x+w)-sin(de)*z+tempx, y+h, sin(de)*(x+w)+cos(de)*z+tempz);
  glEnd();
}
void setOrthographicProjection() {

  // switch to projection mode
  glMatrixMode(GL_PROJECTION);
  // save previous matrix which contains the
  //settings for the perspective projection
  glPushMatrix();
  // reset matrix
  glLoadIdentity();
  // set a 2D orthographic projection
  gluOrtho2D(0, w, 0, h/2);
  // invert the y axis, down is positive
  glScalef(1, -1, 1);
  // mover the origin from the bottom left corner
  // to the upper left corner
  glTranslatef(0, -h/2, 0);
  glMatrixMode(GL_MODELVIEW);
}

void resetPerspectiveProjection() {
  // set the current matrix to GL_PROJECTION
  glMatrixMode(GL_PROJECTION);
  // restore previous settings
  glPopMatrix();
  // get back to GL_MODELVIEW matrix
  glMatrixMode(GL_MODELVIEW);
}

void renderBitmapString(float x, float y, void *font,char *string)
{
  char *c;
  // set position to start drawing fonts
  glRasterPos2f(x, y);
  // loop all the characters in the string
  for (c=string; *c != '\0'; c++) {
    glutBitmapCharacter(font, *c);
  }
}
void initScene() {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  // add fog effect
  /*
  glEnable(GL_FOG);
  {
    GLfloat fogColor[4] = {0.35,0.5,0.5,1.0};

    glFogi(GL_FOG_MODE, GL_EXP);
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogf(GL_FOG_DENSITY, 0.5);
    glHint(GL_FOG_HINT, GL_DONT_CARE);
    glFogf(GL_FOG_START, 0.0);
    glFogf(GL_FOG_END, 10.0);
  }
  glClearColor(0.35,0.5,0.5,1.0);
  */
}

void orientMe(float ang) {
  lx = sin(ang);
  lz = -cos(ang);
}


void moveMeFlat(int i) {
  x = x + i*(lx)*0.01;
  z = z + i*(lz)*0.01;
}

void moveMeVer(int i)
{
  y = y + i*0.01;
}

void renderScene2(int currentWindow) {

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw ground
  glColor3f(0.0f, 0.5f, 0.5f);
  glBegin(GL_QUADS);
  glVertex3f(-200.0f, 0.0f, -200.0f);
  glVertex3f(-200.0f, 0.0f,  200.0f);
  glVertex3f( 200.0f, 0.0f,  200.0f);
  glVertex3f( 200.0f, 0.0f, -200.0f);
  glEnd();
  glBegin(GL_QUADS);
  glVertex3f(-200.0f, 0.0f, -200.0f);
  glVertex3f( 200.0f, 0.0f, -200.0f);
  glVertex3f( 200.0f, 0.0f,  200.0f);
  glVertex3f(-200.0f, 0.0f,  200.0f);
  glEnd();
  // Draw water surface

  glColor3f(0.0f, 0.0f, 0.9f);
  glBegin(GL_QUADS);
  glVertex3f(-200.0f, 16.5f, -200.0f);
  glVertex3f(-200.0f, 16.5f,  200.0f);
  glVertex3f( 200.0f, 16.5f,  200.0f);
  glVertex3f( 200.0f, 16.5f, -200.0f);
  glEnd();
  glBegin(GL_QUADS);
  glVertex3f(-200.0f, 16.5f, -200.0f);
  glVertex3f( 200.0f, 16.5f, -200.0f);
  glVertex3f( 200.0f, 16.5f,  200.0f);
  glVertex3f(-200.0f, 16.5f,  200.0f);
  glEnd();

  outframe++;
  outtime=glutGet(GLUT_ELAPSED_TIME);

  if (outtime - outtimebase > 1000)       // per second
    {
      sprintf(s2,"FPS:%4.2f",outframe*1000.0/(outtime-outtimebase));
      outtimebase = outtime;
      outframe = 0;
    }

  // frequency = 5 Hz with red light
  if(decode == 0)
    {
      if(outtime/100 % 2 == 0)
        glColor3f(1.0f,0.0f,0.0f);
      else    // whiet
        glColor3f(1.0f,1.0f,1.0f);
    }
  // frequency = 2 Hz with red light
  else if(decode == 1)
    {
      if(outtime/100 % 5 == 0)
        glColor3f(1.0f,0.0f,0.0f);
      else    // whiet
        glColor3f(1.0f,1.0f,1.0f);
    }
  // fequency = 5 Hz with green light
  else if(decode == 2)
    {
      if(outtime/100 % 2 == 0)
        glColor3f(0.0f,1.0f,0.0f);
      else    // whiet
        glColor3f(1.0f,1.0f,1.0f);
    }
  // frequency = 2 Hz with green light
  else if(decode == 3)
    {
      if(outtime/100 % 5 == 0)
        glColor3f(0.0f,1.0f,0.0f);
      else    // whiet
        glColor3f(1.0f,1.0f,1.0f);
    }
  glTranslatef(0, 5, 0);
  if(shape)
    glutSolidSphere(0.1f,20,20);
  else
    drawRecRotate(0.0f, -0.15f,-0.15f,-0.25f,0.3f,0.3f,0.3f);
  //      glColor3f(0.0f,0.0f,0.0f);
  //      drawRecRotate(0.0f, -0.15f,-0.15f,-0.25f,0.3f,0.3f,0.3f);
  // show the text

  if (currentWindow == subWindow4)
    {
      glReadPixels(0,0,320,240 ,GL_RGB,GL_UNSIGNED_BYTE,image);
      Image< PixRGB<byte> > im((PixRGB<byte> *)image, 320, 240);

      renderBitmapString(30,15,(void *)font,s);
      frame++;
      atime=glutGet(GLUT_ELAPSED_TIME);

      inStream.push_back(im);

      if (frame%100 == 0)
        {
          avgFR = ((float)frame/(float)(atime/1000));
          sprintf(s2,"FPS:%4.2F",avgFR);
          printf("FPS:%4.2F", avgFR);

          //Use decoder
          test->setupDecoder("Red", true);
          test->runDecoder(inStream, avgFR);
          float hertz = test->calculateHz();
          printf("\n\nFinal hertz calculated is: %f\n\n", hertz);
          inStream.clear();
        }

      glColor3f(0.0,1.0,1.0);
      setOrthographicProjection();
      glPushMatrix();
      glLoadIdentity();
      sprintf(s,"Position:%.1f,%.1f Depth:%.1f ",x,z,(16-y));
      renderBitmapString(30,15,(void *)font,s);
      //              sprintf(s,"F3-Change the signal, F4-Change the shape of the light");
      //              renderBitmapString(30,25,(void *)font,s);
      renderBitmapString(30,35,(void *)font,s2);
      glPopMatrix();
      resetPerspectiveProjection();
    }

  //===========================================
  // test the glReadPixels()
  // unsigned char *image;
  // assign 3 images get from cameras to image[]

  glutSwapBuffers();
}
void renderScene() {
  glutSetWindow(mainWindow);
  glClear(GL_COLOR_BUFFER_BIT);
  glutSwapBuffers();
}

void renderScenesw4() {
  glutSetWindow(subWindow4);
  glLoadIdentity();

  gluLookAt(x, y, z,
            x + lx,y + ly,z + lz,
            0.0f,1.0f,0.0f);

  renderScene2(subWindow4);
}

//=================================
// control the submarine for high level command
void advance(int steps)
{
  deltaMove = steps;
}
/*
  void Turn(string s, double angle)
  {
  if(s=="LEFT")
  deltaAngle = -0.05f * angle;
  else
  deltaAngle = 0.05f * angle;
  }
*/
//=================================
void renderSceneAll() {
  if (deltaMove)
    moveMeFlat(deltaMove);
  if (deltaAngle) {
    fangle += deltaAngle;
    orientMe(fangle);
  }
  if (deltaDepth)
    moveMeVer(deltaDepth);
  renderScenesw4();
}

void processNormalKeys(unsigned char key, int x, int y) {

  if (key == 27)
    exit(0);
}

void pressKey(int key, int x, int y) {

  switch (key) {
  case GLUT_KEY_LEFT : deltaAngle = -0.01f;break;
  case GLUT_KEY_RIGHT : deltaAngle = 0.01f;break;
  case GLUT_KEY_UP : deltaMove = 1;break;
  case GLUT_KEY_DOWN : deltaMove = -1;break;
  case GLUT_KEY_F1 : deltaDepth = 1;break;
  case GLUT_KEY_F2 : deltaDepth = -1;break;
  case GLUT_KEY_F3 :
    decode++;
    if(decode>=4)
      decode = 0;
    break;
  case GLUT_KEY_F4 :
    shape = !shape;
    break;
  }
}

void releaseKey(int key, int x, int y) {

  switch (key) {
  case GLUT_KEY_LEFT :
    if (deltaAngle < 0.0f)
      deltaAngle = 0.0f;
    break;
  case GLUT_KEY_RIGHT :
    if (deltaAngle > 0.0f)
      deltaAngle = 0.0f;
    break;
  case GLUT_KEY_UP :
    if (deltaMove > 0)
      deltaMove = 0;
    break;
  case GLUT_KEY_DOWN :
    if (deltaMove < 0)
      deltaMove = 0;
    break;
  case GLUT_KEY_F1:
    if(deltaDepth > 0)
      deltaDepth =0;
    break;
  case GLUT_KEY_F2:
    if(deltaDepth < 0)
      deltaDepth =0;
    break;
  }
}

#endif // HAVE_GL_GLUT_H

int main(int argc, char **argv)
{
#ifndef HAVE_GL_GLUT_H

  LFATAL("<GL/glut.h> must be installed to use this program");

#else

  mgr.addSubComponent(test);
  mgr.start();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100,100);
  glutInitWindowSize(w,h);
  mainWindow = glutCreateWindow("Sub");
  glutIgnoreKeyRepeat(1);
  glutKeyboardFunc(processNormalKeys);
  glutSpecialFunc(pressKey);
  glutSpecialUpFunc(releaseKey);
  glutReshapeFunc(changeSize);
  glutDisplayFunc(renderScene);
  glutIdleFunc(renderSceneAll);
  subWindow4 = glutCreateSubWindow(mainWindow, 0,0,
                                   w,h);
  glutDisplayFunc(renderScenesw4);
  initScene();
  glutMainLoop();

  return(0);

#endif // HAVE_GL_GLUT_H
}
