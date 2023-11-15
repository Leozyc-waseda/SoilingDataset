/*! @file BeoSub/test-BeoSubSimulator.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubSimulator.C $
// $Id: test-BeoSubSimulator.C 11565 2009-08-09 02:14:40Z rand $

// compile with: -lm -lglut -lGLU -lGL -lX11 -lXext


#include <cmath>
#ifdef HAVE_GL_GLUT_H
#include <GL/glut.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <unistd.h>

#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Image/FilterOps.H"
#include "BeoSub/BeoSubCanny.H"
#include "BeoSub/ColorTracker.H"
#include "CannyModel.H"
#include "BeoSub/BeoSubSim.H"

#ifdef HAVE_GL_GLUT_H

bool start = true;

ModelManager submgr("BeoSubSim Tester");
nub::soft_ref<BeoSubSim> SimSub(new BeoSubSim(submgr));

Angle heading(0.0), deltaHeading(0.0);
float  ratio, dtemp;
Angle pitch(0.0), deltaPitch(0.0);
Angle roll(0.0);
float deltaStrafeX = 0;
float leftThrustValue = 0, rightThrustValue = 0;
bool setForward = false;

float x=-30.0f,y=20.0f,z=10.0f; // init postion of the camera
float targetZ = z, targetX = x;
float subX= 2.0f, subY=3.5f, subZ = 2.0f;
float lx=0.0f,ly=0.0f,lz=-0.1f, ty=0.0f;
float l[] = {0.0,80.0,0.0}; float n[]={0.0,-1.0,0.0}; float e[] = {0.0,-60.0,0.0};
float shadowColor[] = {0.0,0.0,0.0};
GLfloat fogColor[4] = {0.35,0.5,0.5,1.0};
int sw=320,sh=240;
int deltaMove = 0,h=740,w=325, border=5;
int deltaDepth = 0;
int counter=0, actionCount=0;
void* font=GLUT_BITMAP_8_BY_13;
int bitmapHeight=13;
int imgCounter = 0;
GLfloat TCx = 45.0f,TCz = -20.0f;
GLfloat TAx = -15.0f, TAy=4.0f, TAz=0.5f;
int frame,frametime,outframe, outtime,timebase=0,atime, atimebase=0, decode=0;
char s[30];
int mainWindow, subWindow1,subWindow2,subWindow3, subWindow4, subWindow5, subWindow6;
bool set3 = false;
bool fogAct = true, isSphere = true;
void initWindow();

byte *image1 = new byte[sw*sh*3];
byte *image2 = new byte[sw*sh*3];
byte *image3 = new byte[sw*sh*3];

// instantiate a model manager for the shape detector module:
LINFO("CREATING MANAGER");
ModelManager manager("BeoSubCanny Tester");
nub::soft_ref<ColorTracker> ct(new ColorTracker(manager));
  //Test with a circle
rutz::shared_ptr<ShapeModel> shape;
double *darray = (double*)calloc(4, sizeof(double));

Image< PixRGB<byte> > simulateCamera(const byte *data, const int w, const int h);
Image< PixRGB<byte> > ima1, ima2, ima3;

// texture mapping
struct simTexture
{
        unsigned long sizeX;
        unsigned long sizeY;
        char *data;
};
typedef struct simTexture Texture;
int t[50];
// make shadow in the pool
void glShadowProjection(float *l, float *n, float *e)
{
    float d, c, mat[16];
    d = n[0]*l[0] + n[1]*l[1] + n[2]*l[2];
    c = e[0]*n[0] + e[1]*n[1] + e[2]*n[2]-d;

    mat[0] = n[0]*l[0] + c;
    mat[4] = n[1]*l[0];
    mat[8] = n[2]*l[0];
    mat[12]= -l[0]*c - l[0]*d;

    mat[1] = n[0]*l[1];
    mat[5] = n[1]*l[1]+c;
    mat[9] = n[2]*l[1];
    mat[13]= -l[1]*c - l[1]*d;

    mat[2] = n[0]*l[2];
    mat[6] = n[1]*l[2];
    mat[10]= n[2]*l[2]+c;
    mat[14]= -l[2]*c - l[2]*d;

    mat[3] = n[0];
    mat[7] = n[1];
    mat[11]= n[2];
    mat[15]= -d;
    glMultMatrixf(mat);

}

// quick and dirty bitmap loader...for 24 bit bitmaps with 1 plane only.
// See http://www.dcs.ed.ac.uk/~mxr/gfx/2d/BMP.txt for more info.
int ImageLoad(char *filename, Texture *image) {
    FILE *file;
    unsigned long size;                 // size of the image in bytes.
    unsigned long i;                    // standard counter.
    unsigned short int planes;          // number of planes in image (must be 1)
    unsigned short int bpp;             // number of bits per pixel (must be 24)
    char temp;                          // temporary color storage for bgr-rgb conversion.

    // make sure the file is there.
    if ((file = fopen(filename, "rb"))==NULL)
    {
        printf("File Not Found : %s\n",filename);
        return 0;
    }

    // seek through the bmp header, up to the width/height:
    fseek(file, 18, SEEK_CUR);

    // read the width
    if ((i = fread(&image->sizeX, 4, 1, file)) != 1) {
        printf("Error reading width from %s.\n", filename);
        return 0;
    }

    // read the height
    if ((i = fread(&image->sizeY, 4, 1, file)) != 1) {
        printf("Error reading height from %s.\n", filename);
        return 0;
    }
  //  printf("Height of %s: %lu\n", filename, image->sizeY);

    // calculate the size (assuming 24 bits or 3 bytes per pixel).
    size = image->sizeX * image->sizeY * 3;

    // read the planes
    if ((fread(&planes, 2, 1, file)) != 1) {
        printf("Error reading planes from %s.\n", filename);
        return 0;
    }
    if (planes != 1) {
        printf("Planes from %s is not 1: %u\n", filename, planes);
        return 0;
    }

    // read the bpp
    if ((i = fread(&bpp, 2, 1, file)) != 1) {
        printf("Error reading bpp from %s.\n", filename);
        return 0;
    }

    if (bpp != 24) {
        printf("Bpp from %s is not 24: %u\n", filename, bpp);
        return 0;
    }

    // seek past the rest of the bitmap header.
    fseek(file, 24, SEEK_CUR);

    // read the data.
    image->data = (char *) malloc(size);
    if (image->data == NULL) {
        printf("Error allocating memory for color-corrected image data");
        return 0;
    }

    if ((i = fread(image->data, size, 1, file)) != 1) {
        printf("Error reading image data from %s.\n", filename);
        return 0;
    }

    for (i=0;i<size;i+=3) { // reverse all of the colors. (bgr -> rgb)
        temp = image->data[i];
        image->data[i] = image->data[i+2];
        image->data[i+2] = temp;
    }

    // we're done.
    return 1;
}

// Load Bitmaps And Convert To Textures
void LoadGLTextures(char *filename, int i) {
    // Load Texture
    Texture *texture;

    // allocate space for texture
    texture = (Texture *) malloc(sizeof(Texture));
    if (texture == NULL) {
        printf("Error allocating space for image");
        exit(0);
    }

    if (!ImageLoad(filename, texture)) {
        exit(1);
    }

    // Create Texture
    glGenTextures(1, (GLuint *)&t[i]);
    glBindTexture(GL_TEXTURE_2D, t[i]);   // 2d texture (x and y size)

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // scale linearly when image bigger than texture
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); // scale linearly when image smalled than textur

    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    glTexImage2D(GL_TEXTURE_2D, 0, 3, texture->sizeX, texture->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, texture->data);
}

void initWindow2(void)
{
        glClearColor(0.0,0.0,0.0,0.0);
        glShadeModel(GL_FLAT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}
/*
void display4(void)
{
        glutSetWindow(subWindow4);
        glClear(GL_COLOR_BUFFER_BIT);
        //        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, ima1.getArrayPtr());
//        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, image1);
        glutSwapBuffers();
}
void display5(void)
{
        glutSetWindow(subWindow5);
        glClear(GL_COLOR_BUFFER_BIT);
        //        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, ima2.getArrayPtr());
//        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, image2);
        glutSwapBuffers();
}
void display6(void)
{
        glutSetWindow(subWindow6);
        glClear(GL_COLOR_BUFFER_BIT);
        //        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, ima3.getArrayPtr());
//        glDrawPixels(sw,sh,GL_RGB,GL_UNSIGNED_BYTE, image3);
        glutSwapBuffers();
}
*/
void changeSize2(int w1, int h1)
{

        // Prevent a divide by zero, when window is too short
        // (you cant make a window of zero width).
        ratio = 1.0f * w1 / h1;

        // Set the viewport to be the entire window
        glViewport(0, 0, w1, h1);

        // Reset the coordinate system before modifying
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // Set the clipping volume
        gluPerspective(45,ratio,0.1,1000);
        glMatrixMode(GL_MODELVIEW);
}

void changeSize(int w1,int h1) {
        if(h1 == 0)
                h1 = 1;

        w = w1;
        h = h1;
        //mainx =(w-3*border)/2;
        //mainy=2*(h-3*border)/3;

        glutSetWindow(subWindow1);
        glutPositionWindow(border,border);
        glutReshapeWindow(w-border, (h-4*border)/3);
        changeSize2(w-border, (h-4*border)/3);

        glutSetWindow(subWindow2);
        glutPositionWindow(border,2*border+(h-4*border)/3);
        glutReshapeWindow(w-border, (h-4*border)/3);
        changeSize2(w-border, (h-4*border)/3);

        glutSetWindow(subWindow3);
        glutPositionWindow(border,3*border+2*(h-4*border)/3);
        glutReshapeWindow(w-border, (h-4*border)/3);
        changeSize2(w-border, (h-4*border)/3);

        //        glutSetWindow(subWindow4);
        //glutPositionWindow(border+(w-2*border-mainx),border);
        //glutReshapeWindow(w-3*border-mainx, (h-4*border)/3);
        //changeSize2(w-3*border-mainx, (h-4*border)/3);

        //glutSetWindow(subWindow5);
        //glutPositionWindow(border+(w-2*border-mainx),2*border+(h-4*border)/3);
        //glutReshapeWindow(w-3*border-mainx, (h-4*border)/3);
        //changeSize2(w-3*border-mainx, (h-4*border)/3);

        //glutSetWindow(subWindow6);
        //glutPositionWindow(border+(w-2*border-mainx),3*border+2*(h-4*border)/3);
        //glutReshapeWindow(w-3*border-mainx, (h-4*border)/3);
        //changeSize2(w-3*border-mainx, (h-4*border)/3);
}


void drawSnowMan() {}

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

        // init the texturemapping
  //LoadGLTextures("map.bmp", 0);
  /*
    LoadGLTextures("pics/bg00.bmp", 0);                               // Load The Texture(s)
    LoadGLTextures("pics/bg01.bmp", 1);
    LoadGLTextures("pics/bg02.bmp", 2);
        LoadGLTextures("pics/bg03.bmp", 3);
        LoadGLTextures("pics/bg04.bmp", 4);
        LoadGLTextures("pics/bg05.bmp", 5);
        LoadGLTextures("pics/bg06.bmp", 6);
        LoadGLTextures("pics/bg07.bmp", 7);
        LoadGLTextures("pics/bg08.bmp", 8);
        LoadGLTextures("pics/bg09.bmp", 9);
        LoadGLTextures("pics/bg10.bmp", 10);
        LoadGLTextures("pics/bg11.bmp", 11);
        LoadGLTextures("pics/bg12.bmp", 12);
        LoadGLTextures("pics/bg13.bmp", 13);
        LoadGLTextures("pics/0.bmp", 14);
        LoadGLTextures("pics/bg15.bmp", 15);
        LoadGLTextures("pics/bg16.bmp", 16);
        LoadGLTextures("pics/bg17.bmp", 17);
        LoadGLTextures("pics/bg18.bmp", 18);
        LoadGLTextures("pics/bg19.bmp", 19);
        LoadGLTextures("pics/1.bmp", 20);
        LoadGLTextures("pics/b.bmp", 21);
        LoadGLTextures("pics/c.bmp", 22);
        LoadGLTextures("pics/d.bmp", 23);
        LoadGLTextures("pics/e.bmp", 24);
        LoadGLTextures("pics/f.bmp", 25);
        LoadGLTextures("pics/g.bmp", 26);
        LoadGLTextures("pics/h.bmp", 27);
        LoadGLTextures("pics/i.bmp", 28);
        LoadGLTextures("pics/j.bmp", 29);
        LoadGLTextures("pics/k.bmp", 30);
        LoadGLTextures("pics/l.bmp", 31);
        LoadGLTextures("pics/m.bmp", 32);
        LoadGLTextures("pics/n.bmp", 33);
        LoadGLTextures("pics/o.bmp", 34);
        LoadGLTextures("pics/p.bmp", 35);
        LoadGLTextures("pics/q.bmp", 36);
        LoadGLTextures("pics/r.bmp", 37);
        LoadGLTextures("pics/s.bmp", 38);
        LoadGLTextures("pics/2.bmp", 39);

        LoadGLTextures("pics/3.bmp", 40); // not used
        LoadGLTextures("pics/4.bmp", 41);
        LoadGLTextures("pics/5.bmp", 42);
        LoadGLTextures("pics/6.bmp", 43);
        LoadGLTextures("pics/7.bmp", 44);
        LoadGLTextures("pics/8.bmp", 45);
        LoadGLTextures("pics/9.bmp", 46);
*/
        glClearColor(0.0f, 0.0f, 1.0f, 0.0f);   // Clear The Background Color To Blue
        glClearDepth(1.0);                              // Enables Clearing Of The Depth Buffer
        glDepthFunc(GL_LESS);                   // The Type Of Depth Test To Do
        glEnable(GL_DEPTH_TEST);                        // Enables Depth Testing
        glShadeModel(GL_SMOOTH);                        // Enables Smooth Color Shading
        // let the texture behind the shadow
        glEnable(GL_CULL_FACE);

        // add fog effect
        glEnable(GL_FOG);
        {
                glFogi(GL_FOG_MODE, GL_EXP);
                glFogfv(GL_FOG_COLOR, fogColor);
                glFogf(GL_FOG_DENSITY, 0.10);
                glHint(GL_FOG_HINT, GL_DONT_CARE);
                glFogf(GL_FOG_START,1.0);
                glFogf(GL_FOG_END, 5.0);
        }
        glClearColor(0.35,0.5,0.5,1.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();                           // Reset The Projection Matrix
    // Set the clipping volume
    gluPerspective(45,ratio,0.1,1000);
    glMatrixMode(GL_MODELVIEW);
}

void orientMe(Angle theta)
{
        lx = sin(theta.getRadians());
        lz = -cos(theta.getRadians());
}

void moveMeFlat(float i)
{
  x = x + i*(lx)*0.24;
  z = z + i*(lz)*0.24;
}

void moveMeFlatSmall(float i)
{
  x = x + i*(lx)*0.12;
  z = z + i*(lz)*0.12;
}

void moveMeVer(float i)
{
  y = y + i*0.02;
}

void tiltMe(Angle theta)
{
  lz = -cos(theta.getRadians());
  ly = sin(theta.getRadians());

}

void strafe(float val)
{
  x = x + val*lz*0.12;
  z = z + val*lx*0.12;
}

void storeImage(const char* s, Image< PixRGB<byte> > im)
{
  std::cout << s << "image" << imgCounter << '\n';
  imgCounter++;
  Raster::WriteRGB( im, sformat("%simage%d.ppm", s, imgCounter) );
}

void draw(bool shadow)
{
        // draw submarine
/*
        glPushMatrix();
        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(1.0,0.0,0.0);
        glTranslatef(x,y,z);
        glRotatef(180-heading*180.0/3.14,0.0,1.0,0.0);
        drawRecRotate(0.0f,
                     -subX/2.0f, 0.0f,subY,
                      subX, subY, subZ);
        glPopMatrix();
  */
        // draw 5 orange pipes
        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(1.0f, 0.5f, 0.0f);
                drawRecRotate(0.0f,0.0f,0.85f,0.0f,4.0f,0.5f,0.3f);
                drawRecRotate(0.314f,4.3f,0.85f,0.0f,4.0f,0.5f,0.3f);
                drawRecRotate(-0.314f,8.2f,0.85f,1.3f,4.0f,0.5f,0.3f);
                drawRecRotate(0.5f,-4.7f,0.85f,-1.7f,4.0f,0.5f,0.3f);
                drawRecRotate(-0.2f,-9.2f,0.85f,-1.1f,4.0f,0.5f,0.3f);
        // draw the black bin
        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(0.9f,0.9f,0.9f);
        glBegin(GL_POLYGON);
                glVertex3f(-1.85f, 1.2f,-0.5f);

                glVertex3f(-1.78f, 1.2f,-0.25f);
                glVertex3f(-1.965f,1.2f, 0.0f);
                glVertex3f(-1.75f,1.2f, 0.15f);
                glVertex3f(-1.755f,1.2f, 0.35f);
                glVertex3f(-1.878f,1.2f, 0.65f);
                glVertex3f(-1.679f,1.2f, 0.95f);

                glVertex3f(-1.65f, 1.2f, 1.0f);

                glVertex3f(-1.35f,1.2f, 1.42f);
                glVertex3f(-1.12f,1.2f, 1.732f);
                glVertex3f(-1.03f,1.2f, 1.626f);

                glVertex3f( 0.0f, 1.2f, 1.34f);

                glVertex3f( 0.412f,1.2f, 0.35f);
                glVertex3f( 0.532f,1.2f, 0.24f);
                glVertex3f( 0.325f,1.2f, 0.0f);
                glVertex3f( 0.65f,1.2f,-0.15f);
                glVertex3f( 0.31f, 1.2f,-0.35f);
                glVertex3f( 0.46f,1.2f,-0.44f);

                glVertex3f( 0.60f, 1.2f,-0.5f);

                glVertex3f(-0.11f,1.2f,-0.965f);
                glVertex3f(-0.25f,1.2f,-0.6f);
                glVertex3f(-0.37f,1.2f,-0.875f);
                glVertex3f(-0.65f,1.2f,-0.72f);
                glVertex3f(-0.84f,1.2f,-0.966f);
                glVertex3f(-0.99f,1.2f,-0.77f);
                glVertex3f(-1.23f,1.2f,-0.677f);

        glEnd();

        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(0.0f,0.0f,0.0f);
        drawRecRotate(0.0f, -1.25f,1.2f,-0.25f,1.0f,1.0f,0.1f);

        // draw the octagon
        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(1.0f,0.5f,0.0f);
                drawRecRotate(0.0f,0.0f+TCx,16.0f,0.0f+TCz,6.21f,0.5f,0.3f);  // -
                drawRecRotate(0.785f,6.21f+TCx,16.0f,0.0f+TCz,6.21f,0.5f,0.3f);
                drawRecRotate(1.57f,10.6f+TCx,16.0f,4.39f+TCz,6.21f,0.5f,0.3f); // |
                drawRecRotate(-0.785f,-4.39f+TCx,16.0f,4.39f+TCz,6.21f,0.5f,0.3f); // /
                drawRecRotate(1.57f,-3.89f+TCx,16.0f,4.39f+TCz,6.21f,0.5f,0.3f); // |
                drawRecRotate(0.785f,-3.89f+TCx,16.0f,10.3f+TCz,6.21f,0.5f,0.3f);
                drawRecRotate(-0.785f,6.0f+TCx,16.0f,14.5f+TCz,6.21f,0.5f,0.3f); // /
                drawRecRotate(0.0f,0.0f+TCx,16.0f,14.5f+TCz,6.21f,0.5f,0.3f);  // -

                drawRecRotate(0.0f,0.85f+TCx, 16.0f, 2.75f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(0.785f,5.0f+TCx, 16.0f, 2.75f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(2.355f,1.1f+TCx, 16.0f, 3.0f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(1.57f,7.917f+TCx, 16.0f, 5.677f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(1.57f,-1.8f+TCx, 16.0f, 5.677f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(0.785f,-1.8f+TCx, 16.0f, 9.5f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(2.355f,7.917f+TCx, 16.0f, 9.7f+TCz,4.14f,0.5f,0.3f);
                drawRecRotate(0.0f,0.85f+TCx, 16.0f, 12.25f+TCz,4.14f,0.5f,0.3f);
        outtime = glutGet(GLUT_ELAPSED_TIME);
        // draw the light for task a
        if(shadow)
                glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
        {
        if (outtime % 3 ==0)
                glColor3f(1.0,0.0,0.0);
        else
                glColor3f(1.0,1.0,1.0);
        }
        glTranslatef(TAx, TAy, TAz);
        glutSolidSphere(0.25f,20,20);

        // draw the gate

        if(shadow)
                   glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
                glColor3f(1.0f,1.0f,1.0f);
                drawRecRotate(0.0f, -10.0f,TAy+1.0,10.0f,0.5f,0.5f,6.0f);
                drawRecRotate(0.0f, -10.0f,TAy+1.0,0.0f,0.5f,0.5f,6.0f);

        // draw the flashing light box
        // frequency = 5 Hz with the red light
        if(shadow)
                 glColor3f(shadowColor[0],shadowColor[1],shadowColor[2]);
        else
        {
        if(decode == 0)
        {
          if (outtime % 2 ==0)
                glColor3f(1.0,0.0,0.0);
          else
                glColor3f(1.0,1.0,1.0);
        }
        // frequency = 2 Hz with the red light
        else if(decode == 1)
        {
          if (outtime % 5 ==0)
                glColor3f(1.0,0.0,0.0);
          else
                glColor3f(1.0,1.0,1.0);
         }
        // frequency = 5 Hz with the green light
        else if(decode == 2)
        {
          if (outtime % 2 ==0)
                glColor3f(0.0,1.0,0.0);
          else
                glColor3f(1.0,1.0,1.0);
        }
        // frequency - 2 Hz with the green light
        else
        {
          if (outtime % 5 ==0)
                glColor3f(0.0,1.0,0.0);
          else
                glColor3f(1.0,1.0,1.0);
        }
        }
        glTranslatef(-9.15,7.0,9.0);
        if(isSphere)
                glutSolidSphere(0.1f, 20,20);
        else
                drawRecRotate(0.0f,0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
}
void renderScene2(int currentWindow) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);         // Clear The Screen And The Depth Buffer
    glLightfv(GL_LIGHT0, GL_POSITION, l);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);

    glColor3f(1.0,1.0,0.0);
    glBegin(GL_POINTS);
    glVertex3f(l[0],l[1],l[2]);
    glEnd();

    // put the texture on the follor
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    // put pictures in the size of 20*20 in x axis

    //TEST for large map
    glBindTexture(GL_TEXTURE_2D, t[0]);
    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f); glVertex3f(-100.0f+(80), -1.5f,-40.0f);//Bottom Left
    glTexCoord2f(1.0f, 0.0f); glVertex3f(-100.0f+(80), -1.5f,-20.0f);//Bottom Right
    glTexCoord2f(1.0f, 1.0f); glVertex3f(-80.0f+(80), -1.5f,-20.0f);//Top Right
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-80.0f+(80), -1.5f,-40.0f);//Top Left

/*
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-100.0f+(80), -1.5f,-40.0f+(200));//Bottom Left
    glTexCoord2f(1.0f, 0.0f); glVertex3f(-100.0f+(80), -1.5f,-20.0f+(200));//Bottom Right
    glTexCoord2f(1.0f, 1.0f); glVertex3f(-80.0f+(80), -1.5f,-20.0f+(200));//Top Right
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-80.0f+(80), -1.5f,-40.0f+(200));//Top Left
    glEnd();
  */  /*
    for(int i=0 ; i<10 ; i++)
      {
        for(int j=0 ; j<4 ; j++)
          {
            glBindTexture(GL_TEXTURE_2D, t[i*4+j]);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(-100.0f+(i*20), -1.5f,-40.0f+(j*20));//Bottom Left
            glTexCoord2f(1.0f, 0.0f); glVertex3f(-100.0f+(i*20), -1.5f,-20.0f+(j*20));//Bottom Right
            glTexCoord2f(1.0f, 1.0f); glVertex3f(-80.0f+(i*20), -1.5f,-20.0f+(j*20));//Top Right
            glTexCoord2f(0.0f, 1.0f); glVertex3f(-80.0f+(i*20), -1.5f,-40.0f+(j*20));//Top Left
            glEnd();
          }
      }
    */
    //glDisable(GL_TEXTURE_2D);

        // show fog or not
        if(fogAct)
                glEnable(GL_FOG);
        else
                glDisable(GL_FOG);
        // Draw ground
        glColor3f(0.0f, 0.5f, 0.5f);
        glBegin(GL_QUADS);
        glNormal3f(0.0,1.0,0.0);
                glVertex3f(-100.0f, e[1]-0.1f, -100.0f);
                glVertex3f(-100.0f, e[1]-0.1f,  100.0f);
                glVertex3f( 100.0f, e[1]-0.1f,  100.0f);
                glVertex3f( 100.0f, e[1]-0.1f, -100.0f);
        glEnd();
        glBegin(GL_QUADS);
                glVertex3f(-100.0f, e[1]-0.1f, -100.0f);
                glVertex3f( 100.0f, e[1]-0.1f, -100.0f);
                glVertex3f( 100.0f, e[1]-0.1f,  100.0f);
                glVertex3f(-100.0f, e[1]-0.1f,  100.0f);
        glEnd();

        // Draw water surface
        glColor3f(0.0f, 0.0f, 0.9f);
        glBegin(GL_QUADS);
        glNormal3f(0.0,-1.0,0.0);
                glVertex3f(-100.0f, 16.5f, -100.0f);
                glVertex3f(-100.0f, 16.5f,  100.0f);
                glVertex3f( 100.0f, 16.5f,  100.0f);
                glVertex3f( 100.0f, 16.5f, -100.0f);
        glEnd();
        glBegin(GL_QUADS);
                glVertex3f(-200.0f, 16.5f, -200.0f);
                glVertex3f( 200.0f, 16.5f, -200.0f);
                glVertex3f( 200.0f, 16.5f,  200.0f);
                glVertex3f(-200.0f, 16.5f,  200.0f);
        glEnd();

        // draw the object that cast th shadow
        glPushMatrix();
//        glEnable(GL_LIGHTING);
        draw(false);
        glPopMatrix();

        // draw the shadow
        glPushMatrix();
        glShadowProjection(l,e,n);
        glDisable(GL_LIGHTING);
        glColor3f(0.5,0.5,0.5);
        draw(true);
        glPopMatrix();

        //********************
//          glPopMatrix();
        // show the text
        if (currentWindow == subWindow1)
        {
                /*
                glColor3f(0.0,1.0,1.0);
                setOrthographicProjection();
                glPushMatrix();
                glLoadIdentity();
                sprintf(s,"Position:%.1f,%.1f Depth:%.1f ",x,z,(16-y));
                renderBitmapString(30,15,(void *)font,s);
                dtemp = heading/3.14*180;
                while(dtemp>360)
                        dtemp = dtemp -360;
                while(dtemp<0)
                        dtemp = dtemp + 360;
                sprintf(s,"Heading:%.1f",dtemp);
                renderBitmapString(10,10,(void *)font,s);
                renderBitmapString(150,150,(void *)font,"For usrc: Navigate the world by arrow keys,");
                renderBitmapString(30,0,(void *)font,"F1 - Rise, F2 - Dive, ESC - Quit");
                glPopMatrix();
                resetPerspectiveProjection();
                */
//              printf("D:%.1f\n", (16-y));
          frame++;
          frametime = glutGet(GLUT_ELAPSED_TIME);
          if(frametime - timebase > 1000) // per second
            {
              printf("FPS:%4.2f\n", frame * 1000.0 / (frametime- timebase));
              timebase = frametime;
              frame = 0;
            }

        }
//===========================================
// test the glReadPixels()
// unsigned char *image;
// assign 3 images get from cameras to image[]
        if(currentWindow==subWindow1)
        {
          glReadPixels(0,0,sw,sh,GL_RGB,GL_UNSIGNED_BYTE,image1);
          ima1 = simulateCamera(image1, sw, sh);
        }
        if(currentWindow==subWindow2)// && outtime/10 % 10 ==0)
        {
          glReadPixels(0,0,sw,sh,GL_RGB,GL_UNSIGNED_BYTE,image2);
          ima2 = simulateCamera(image2, sw, sh);
                // test BeoSubCanny
          if(!start)
          {
            /*
                test->setupCanny("Red", ima2, false);
                darray[1] = 100.0;
                   darray[2] = 80.0;
                darray[3] = 50.0f;
                shape.reset(new CircleShape(25.0, darray);

                bool find = test->runCanny(shape);
                double *d = shape->getDimensions();
                if(find)
                printf("::::::::::::::%f\n",d[1]);
            */

          }

        }
        if(currentWindow==subWindow3)// && outtime/10 % 10 ==0)
        {
          glReadPixels(0,0,sw,sh,GL_RGB,GL_UNSIGNED_BYTE,image3);
          ima3 = simulateCamera(image3, sw, sh);
        }


        glutSwapBuffers();
}
void renderScene() {
        glutSetWindow(mainWindow);
        glClear(GL_COLOR_BUFFER_BIT);
        glutSwapBuffers();
}

void renderScenesw1() {
        glutSetWindow(subWindow1);
        glLoadIdentity();
        gluLookAt(x, y+0.5, z,
                      x,y + 1 +ly+ ty,z,
                          -lx,0,-lz);
        renderScene2(subWindow1);
}

void renderScenesw2() {
        glutSetWindow(subWindow2);
        glLoadIdentity();
        gluLookAt(x, y, z,
                      x + lx,y + ly + ty,z + lz,
                          0.0f,1.0f,0.0f);
        renderScene2(subWindow2);
}

void renderScenesw3() {
        glutSetWindow(subWindow3);
        glLoadIdentity();
        gluLookAt(x, y-0.5, z,
                      x,y - 1 +ly+ ty,z,
                          lx,0,lz);
        renderScene2(subWindow3);
}
/*
void *taska(void *ptr)
{
  printf("taska");
  turn(180);
  return NULL;
}
*/
//================================
//define the actions here
void advanceByTime(int steps, bool direction)
{
        int d;
        if(direction)
                d = 1;
        else
                d = -1;

        do
        {
                atime = glutGet(GLUT_ELAPSED_TIME);
                x = x + d*(lx)*0.25;
                z = z + d*(lz)*0.25;
                // render the screen
                renderScenesw1();
                renderScenesw2();
                renderScenesw3();
                //  display4();
                // display5();
                //display6();
        }while(atime - atimebase < steps*1000);
        atimebase = atime;

}



/*void turn(double dangle)
{
        double count = 0.0;
        double diff;
        dangle = dangle/180 * 3.14;

        if(dangle >= 0)
        {
          diff = 0.0157;
          do
          {
                heading += diff;
                lx = sin(heading.getRadians());
                lz = -cos(heading.getRadians());
                // render the screen
                renderScenesw1();
                renderScenesw2();
                renderScenesw3();
                display4();
                display5();
                display6();
                count += diff;
          }while(count <= dangle);
        }
        else
        {
          diff = -0.0157;
          do
          {
                heading += diff;
                lx = sin(heading);
                lz = -cos(heading);
                // render the screen
                renderScenesw1();
                renderScenesw2();
                renderScenesw3();
                display4();
                display5();
                display6();
                count += diff;
          }while(count >= dangle);
        }
        //atimebase = atime;

}*/
//===============================
// define the action for task a
/*
  void taska()
  {
  float xpos, ypos;
  atimebase = glutGet(GLUT_ELAPSED_TIME);

  // add the component
  if(actionCount == 11)
  {
  manager.addSubComponent(ct);
  manager.start();
  }
  // dive first
  else if(actionCount == 12)
  dive( y - 5, true);
  // assume we are close to the target
  else if(actionCount == 13)
  {
  ct->setupTracker("Red", ima2, false);
  while(!ct->runTracker(50.0, xpos, ypos))
  {
  turn(5);
  ct->setupTracker("Red", ima2,false);
  }

  }


  actionCount ++;
  }
*/
//=================================
//acions commands in here
void renderSceneAll() {
  /*

  if(deltaHeading.getVal() > 0 && !(SimSub->targetReached())) {
    heading += deltaHeading;
    orientMe(heading);
    std::cout << "Heading: " << SimSub->getHeading().getVal() << '\n';
    deltaHeading = 1;
  }

  if(deltaStrafeX) {
    strafe(deltaStrafeX);
    std::cout << "sub x: " << x << " sub z: " << z << '\n';
    deltaStrafeX = 0;
  */


  //if(!setDive) {
  //  SimSub->diveAbs(3);
  //}

  if(!setForward) {
    //  SimSub->strafeRel(5);

    SimSub->getThrusters(leftThrustValue, rightThrustValue);

    if(leftThrustValue != 0) {
      targetZ = -(z + SimSub->getRelDist() * 3.2808 * lz);
      targetX = x - SimSub->getRelDist() * 3.2808 * lx;
      setForward = true;
    }
    else if(SimSub->isStrafing()) {
      targetZ = -(z + SimSub->getRelDist() * 3.2808 * lx);
      targetX = x - SimSub->getRelDist() * 3.2808 * lz;
      setForward = true;
    }

  }


  std::cout << "goalValueZ: " << targetZ * .3048 << "goalValueX: " << targetX * .3048<< '\n';
  std::cout << z*.3048 << " " <<  x*.3048 << '\n';
  //advance forward/reverse
  if(leftThrustValue != 0 && fabs(targetZ - z) >= 0.2 && fabs(targetX - x) >=0.2)
    moveMeFlat(3.2808f * leftThrustValue);

  if(SimSub->isStrafing() && fabs(targetZ - z) >= 0.2 && fabs(targetX - x) >= 0.2)
    strafe(3.2808);

  if(fabs(targetZ - z) < 0.2 && fabs(targetX - x) < 0.2) {
    std::cout << "done advancing\n";
    SimSub->thrust(0.0F, 0.0F);
    SimSub->setStrafing();
  }



  //this code works turning right/left
  if(fabs(SimSub->getTargetAttitude().heading.getRadians() -
     SimSub->getCurrentAttitude().heading.getRadians()) > 0.2) {
    std::cout << "turning\n";
    std::cout << "heading target: " << SimSub->getTargetAttitude().heading.getVal() << '\n';
    if(SimSub->getTargetAttitude().heading.getRadians() > 0)
      heading+=1;
    else
      heading-=1;

    orientMe(heading);
  }
  else
    std::cout << "done turning\n";


  //this code works for diving/surfacing
  if(fabs(SimSub->getTargetAttitude().depth - SimSub->getCurrentAttitude().depth) > 0.2) {
    std::cout << "diving\n";
    std::cout << "depth target: " << SimSub->getTargetAttitude().depth << "m\n";

    if(SimSub->getTargetAttitude().depth > SimSub->getCurrentAttitude().depth)
      moveMeVer(-3.2808f);
    else
      moveMeVer(3.2808f);
  }
  else
    std::cout << "done diving\n";

  if(SimSub->getUpGrabFlag()) {
    storeImage("up", ima1);
    SimSub->setUpGrabFlag();
  }

  if(SimSub->getFrontGrabFlag()) {
    storeImage("front", ima2);
    SimSub->setFrontGrabFlag();
  }

  if(SimSub->getDownGrabFlag()) {
    storeImage("up", ima3);
    SimSub->setDownGrabFlag();
  }

  //this code works
  SimSub->updateCompass(heading, pitch, roll);
  SimSub->updateDepth((20 - y) * 0.3048f);
  SimSub->updatePosition(z * 0.3048f, x * 0.3048f);
  std::cout << "heading: " << SimSub->getHeading().getVal()
       << " depth: " << SimSub->getCurrentAttitude().depth << "m\n";
  std::cout << "z: " << SimSub->getCurrentZ() << "m x: " << SimSub->getCurrentX() << "m\n";



  renderScenesw1();
  renderScenesw2();
  renderScenesw3();
  //  display4();
  //display5();
  //display6();
}

void processNormalKeys(unsigned char key, int x, int y) {
        if (key == 27)
                exit(0);
}

void mouseButton(int button, int state, int x, int y) {
  if(button == GLUT_LEFT_BUTTON)
    SimSub->diveRel(3);
  else if(button == GLUT_RIGHT_BUTTON)
    SimSub->turnRel(90);
  else if(button == GLUT_MIDDLE_BUTTON)
    SimSub->TaskA();
}

// these keys will be used to choose which task to do,
// i.e. F1 is TaskA()
void pressKey(int key, int x, int y) {
  switch (key) {
        case GLUT_KEY_LEFT : {
          std::cout << "ADVANCE!\n";
          SimSub->advanceRel(5);  break;
        }
        case GLUT_KEY_RIGHT : SimSub->diveRel(5) ;break;
        case GLUT_KEY_UP : SimSub->turnRel(5); break;
        case GLUT_KEY_DOWN : deltaMove = -1;break;
        case GLUT_KEY_F1 : deltaDepth = 1;break;
        case GLUT_KEY_F2 : deltaDepth = -1;break;
        case GLUT_KEY_F3 : deltaStrafeX = 1; break;
          //decode++;
          //if(decode >= 4)
          //decode = 0;
          //break;
  case GLUT_KEY_F4 : deltaStrafeX = -1;
    //                    fogAct = false;
                          break;
  case GLUT_KEY_F5 :
                  isSphere = !isSphere;
                        break;
  case GLUT_KEY_F6 :
    start = !start;
    manager.addSubComponent(ct);
    manager.start();

    break;
  case GLUT_KEY_F7 : storeImage("up", ima1); break;
  case GLUT_KEY_F8 : storeImage("front", ima2); break;
  case GLUT_KEY_F9 : storeImage("down",ima3); break;
  case GLUT_KEY_PAGE_UP : deltaPitch = 0.05f; break;
  case GLUT_KEY_PAGE_DOWN : deltaPitch = -0.05f; break;
  }
}

void releaseKey(int key, int x, int y) {

     switch (key) {
                case GLUT_KEY_LEFT :
                  SimSub->advanceRel(5);
                  break;
                /* case GLUT_KEY_RIGHT :
                        if (deltaHeading > 0.0f)
                          deltaHeading = 0.0f;
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
                                case GLUT_KEY_F3:
          if(deltaStrafeX < 0)
            deltaStrafeX = 0;
          break;
        case GLUT_KEY_F4:
          if(deltaStrafeX > 0)
            deltaStrafeX = 0;
          break;
        case GLUT_KEY_F5:
          if(deltaStrafeZ < 0)
          deltaStrafeY
                case GLUT_KEY_F4 :
                        fogAct = true;
                        break;
                case GLUT_KEY_PAGE_UP :
                        if(deltaPitch > 0)
                                deltaPitch = 0;
                break;
                case GLUT_KEY_PAGE_DOWN :
                         if(deltaPitch < 0)
                                deltaPitch = 0;
                                break;*/
                        }
}

// assume RGB RGB RGB ...
// 3 bytes per pixel
// data is 'w' pixels wide by 'h' pixels high
// hence data should have at least w*h*3 bytes of memory allocated
Image< PixRGB<byte> >
simulateCamera(const byte *data, const int aw, const int ah)
{
  // build an image from the raw data:
  Image< PixRGB<byte> > im((PixRGB<byte> *)data, aw, ah);

  // add some small amount of speckle noise to the image:
  //inplaceColorSpeckleNoise(im, 1000);

  // blur the image:
  //im = lowPass9(lowPass9(lowPass9(im)));
  //im = lowPass9(im);

  // open a window if not open yet:
 // if (xw == NULL) xw = new XWindow(Dims(w, h));

  // show image in the window:
 // xw->drawImage(im);

  return im;
}
/*
void *control(bool dummy)
{
  printf("\n\ngot in\n\n");

  //===============================
  // define the action for task a

  //TASK A

}
*/

#endif // HAVE_GL_GLUT_H

int main(int argc, char **argv)
{
#ifndef HAVE_GL_GLUT_H

  LFATAL("<GL/glut.h> must be installed to use this program");

#else

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
        glutInitWindowPosition(0,0);
        glutInitWindowSize(w,h);
        mainWindow = glutCreateWindow("Sub");
        glutIgnoreKeyRepeat(1);
        glutKeyboardFunc(processNormalKeys);
        glutSpecialFunc(pressKey);
        glutSpecialUpFunc(releaseKey);
        glutMouseFunc(mouseButton);
        glutReshapeFunc(changeSize);
        glutDisplayFunc(renderScene);
        glutIdleFunc(renderSceneAll);

        glEnable(GL_NORMALIZE);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHT0);
        glEnable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION);

        glLoadIdentity();
        gluPerspective(60.0f,1.0,1.0,400.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0,0.0,-150.0);


        //setup windows
        subWindow1 = glutCreateSubWindow(mainWindow, border,border,
                                         w-border, (h-4*border)/3);
        glutDisplayFunc(renderScene);
        initScene();

        subWindow2 = glutCreateSubWindow(mainWindow, border,2*border+(h-4*border)/3,
                                         w-border, (h-4*border)/3);
        glutDisplayFunc(renderScenesw2);
        initScene();

        subWindow3 = glutCreateSubWindow(mainWindow, border,3*border+2*(h-4*border)/3,
                                         w-border, (h-4*border)/3);
        glutDisplayFunc(renderScenesw3);
        initScene();

        //subWindow4 = glutCreateSubWindow(mainWindow,w-border-mainx, border,
        //                           w-3*border-mainx, (h-4*border)/3);
        //glutDisplayFunc(display4);
        //initWindow2();

        //subWindow5 = glutCreateSubWindow(mainWindow,w-border-mainx,2*border+(h-4*border)/3,
        //                                 w-3*border-mainx, (h-4*border)/3);
        //glutDisplayFunc(display5);
        //initWindow2();
        //subWindow6 = glutCreateSubWindow(mainWindow,w-border-mainx,3*border+2*(h-4*border)/3,
        //                         w-3*border-mainx, (h-4*border)/3);
        //glutDisplayFunc(display6);
        //initWindow2();

        glutMainLoop();
        return 1;

#endif // HAVE_GL_GLUT_H

}

