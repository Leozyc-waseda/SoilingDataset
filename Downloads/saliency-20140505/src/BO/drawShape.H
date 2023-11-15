#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void square(float x, float y,float color, float times, float sx, float sy)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.2305)*times+sx, y*times+sy);
  glVertex2d((x+0.2305)*times+sx, (y+0.2305)*times+sy);
  glVertex2d(x*times+sx, (y+0.2315)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.2305)*times+sx, y*times+sy);
  glVertex2d((x+0.2305)*times+sx, (y+0.2305)*times+sy);
  glVertex2d(x*times+sx, (y+0.2305)*times+sy);
  glEnd();

  glPopMatrix();
}

void edge_row(float x, float y,float color, float times, float sx, float sy) //sx,sy : center
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glPopMatrix();
}

void edge_col(float x, float y,float color, float times, float sx, float sy)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) color, (float) color);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glPopMatrix();
}

void blue_edge_row(float x, float y,float color, float times, float sx, float sy) //sx,sy : center
     /* BO right */
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) 0.0, (float) 0.0, (float) color);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) 0.0, (float) 0.0, (float) color);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glPopMatrix();
}

void blue_edge_col(float x, float y,float color, float times, float sx, float sy)
     /* BO right */
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) 0.0, (float) 0.0, (float) color);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) 0.0, (float) 0.0, (float) color);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glPopMatrix();
}

void red_edge_row(float x, float y,float color, float times, float sx, float sy) //sx,sy : center
     /* BO left */
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) 0.0, (float) 0.0);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) 0.0, (float) 0.0);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, y*times+sy);
  glVertex2d((x+0.29)*times+sx, (y+0.08)*times+sy);
  glVertex2d(x*times+sx, (y+0.08)*times+sy);
  glEnd();

  glPopMatrix();
}

void red_edge_col(float x, float y,float color, float times, float sx, float sy)
     /* BO left */
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) 0.0, (float) 0.0);

  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d((float) color, (float) 0.0, (float) 0.0);
  glVertex2d(x*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, y*times+sy);
  glVertex2d((x+0.08)*times+sx, (y+0.3)*times+sy);
  glVertex2d(x*times+sx, (y+0.3)*times+sy);
  glEnd();

  glPopMatrix();
}

void draw_Square1(int type, float x, float y, float color, float times) //New Shape
{
  switch(type){
  case 0: //New shape is Right
    {
      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(-0.45,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);
    }break;
  case 1:{ //Nwe shape is Left
    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.9,0.0,color,times,x,y);

    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.9,0.225,color,times,x,y);

    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.9,-0.225,color,times,x,y);

    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.9,-0.45,color,times,x,y);

    square(0.225,-0.675,color,times,x,y);
    square(0.0,-0.675,color,times,x,y);
    square(-0.225,-0.675,color,times,x,y);
    square(-0.45,-0.675,color,times,x,y);
    square(-0.675,-0.675,color,times,x,y);
    square(-0.9,-0.675,color,times,x,y);

    square(0.225,-0.9,color,times,x,y);
    square(0.0,-0.9,color,times,x,y);
    square(-0.225,-0.9,color,times,x,y);
    square(-0.45,-0.9,color,times,x,y);
    square(-0.675,-0.9,color,times,x,y);
    square(-0.9,-0.9,color,times,x,y);
  }break;
  case 2:{ //Reverse Newshape Right
    square(-0.45,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.675,-0.225,color,times,x,y);

    square(-0.45,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.675,-0.45,color,times,x,y);

    square(-0.45,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.675,0.0,color,times,x,y);

    square(-0.45,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.675,0.225,color,times,x,y);

    square(-0.45,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
    square(0.675,0.45,color,times,x,y);

    square(-0.45,0.675,color,times,x,y);
    square(-0.225,0.675,color,times,x,y);
    square(0.0,0.675,color,times,x,y);
    square(0.225,0.675,color,times,x,y);
    square(0.45,0.675,color,times,x,y);
    square(0.675,0.675,color,times,x,y);
  }break;
  case 3:{ //Reverse Newshape Left
    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.9,-0.225,color,times,x,y);

    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.9,-0.45,color,times,x,y);

    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.9,0.0,color,times,x,y);

    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.9,0.225,color,times,x,y);

    square(0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);
    square(-0.675,0.45,color,times,x,y);
    square(-0.9,0.45,color,times,x,y);

    square(0.225,0.675,color,times,x,y);
    square(0.0,0.675,color,times,x,y);
    square(-0.225,0.675,color,times,x,y);
    square(-0.45,0.675,color,times,x,y);
    square(-0.675,0.675,color,times,x,y);
    square(-0.9,0.675,color,times,x,y);
  }break;
  }

}

void draw_Square2(int type, float x, float y, float color, float times) //New Shape
{
  switch(type){
  case 0: //New shape is Right
    {
      square(-0.9,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(0.225,0.675,color,times,x,y);

      square(-0.9,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
    }break;
  case 1:{ //Nwe shape is Left
    square(0.675,0.675,color,times,x,y);
    square(0.45,0.675,color,times,x,y);
    square(0.225,0.675,color,times,x,y);
    square(0.0,0.675,color,times,x,y);
    square(-0.225,0.675,color,times,x,y);
    square(-0.45,0.675,color,times,x,y);

    square(0.675,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);

    square(0.675,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);

    square(0.675,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);

    square(0.675,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);

    square(0.675,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
  }break;
  case 2:{ //Reverse Newshape Right
    square(-0.9,-0.9,color,times,x,y);
    square(-0.675,-0.9,color,times,x,y);
    square(-0.45,-0.9,color,times,x,y);
    square(-0.225,-0.9,color,times,x,y);
    square(0.0,-0.9,color,times,x,y);
    square(0.225,-0.9,color,times,x,y);

    square(-0.9,-0.675,color,times,x,y);
    square(-0.675,-0.675,color,times,x,y);
    square(-0.45,-0.675,color,times,x,y);
    square(-0.225,-0.675,color,times,x,y);
    square(0.0,-0.675,color,times,x,y);
    square(0.225,-0.675,color,times,x,y);

    square(-0.9,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);

    square(-0.9,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);

    square(-0.9,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);

    square(-0.9,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
  }break;
  case 3:{ //Reverse Newshape Left
    square(0.675,-0.9,color,times,x,y);
    square(0.45,-0.9,color,times,x,y);
    square(0.225,-0.9,color,times,x,y);
    square(0.0,-0.9,color,times,x,y);
    square(-0.225,-0.9,color,times,x,y);
    square(-0.45,-0.9,color,times,x,y);

    square(0.675,-0.675,color,times,x,y);
    square(0.45,-0.675,color,times,x,y);
    square(0.225,-0.675,color,times,x,y);
    square(0.0,-0.675,color,times,x,y);
    square(-0.225,-0.675,color,times,x,y);
    square(-0.45,-0.675,color,times,x,y);

    square(0.675,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);

    square(0.675,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);

    square(0.675,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);

    square(0.675,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
  }break;
  }

}


void draw_New1(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);

      square(0.225,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,-0.0,color,times,x,y);
      square(0.0,-0.0,color,times,x,y);
      square(0.225,-0.0,color,times,x,y);
      square(0.45,-0.0,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);

      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);

      square(-0.675,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
    }break;
  case 1:
    {
      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);

      square(-0.45,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,-0.0,color,times,x,y);
      square(-0.225,-0.0,color,times,x,y);
      square(-0.45,-0.0,color,times,x,y);
      square(-0.675,-0.0,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);

      square(0.45,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);

      square(0.45,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
    }break;
  case 2:{ // Reverse New shape Right
    square(-0.45,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);

    square(0.225,-0.675,color,times,x,y);
    square(0.45,-0.675,color,times,x,y);

    square(-0.45,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);

    square(-0.45,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);

    square(-0.675,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);

    square(-0.675,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
  }break;
  case 3:{ //Reverse New shape Left
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);

    square(-0.45,-0.675,color,times,x,y);
    square(-0.675,-0.675,color,times,x,y);

    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);

    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);

    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);

    square(0.45,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);
    square(-0.675,0.45,color,times,x,y);
  }break;
  }
}

void draw_New2(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      square(-0.9,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);

      square(-0.9,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
    }break;
  case 1:
    {
      square(0.675,0.675,color,times,x,y);
      square(0.45,0.675,color,times,x,y);

      square(0.675,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);

      square(0.675,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);

      square(0.675,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);

      square(0.675,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);

      square(0.675,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
    }break;
  case 2:{ // Reverse New shape Right
    square(-0.9,-0.9,color,times,x,y);
    square(-0.675,-0.9,color,times,x,y);

    square(-0.9,-0.675,color,times,x,y);
    square(-0.675,-0.675,color,times,x,y);
    square(-0.45,-0.675,color,times,x,y);
    square(-0.225,-0.675,color,times,x,y);
    square(0.0,-0.675,color,times,x,y);

    square(-0.9,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);

    square(-0.9,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);

    square(-0.9,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);

    square(-0.9,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
  }break;
  case 3:{ //Reverse New shape Left
    square(0.675,-0.9,color,times,x,y);
    square(0.45,-0.9,color,times,x,y);

    square(0.675,-0.675,color,times,x,y);
    square(0.45,-0.675,color,times,x,y);
    square(0.225,-0.675,color,times,x,y);
    square(0.0,-0.675,color,times,x,y);
    square(-0.225,-0.675,color,times,x,y);

    square(0.675,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);

    square(0.675,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);

    square(0.675,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);

    square(0.675,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
  }break;
  }
}

void draw_New3(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);
    }break;
  case 1:
    {
      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.9,-0.675,color,times,x,y);

      square(-0.675,-0.9,color,times,x,y);
      square(-0.9,-0.9,color,times,x,y);
    }break;
  case 2:{ // Reverse New shape Right
    square(-0.45,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.675,-0.45,color,times,x,y);

    square(-0.45,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.675,-0.225,color,times,x,y);

    square(-0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.675,0.0,color,times,x,y);

    square(-0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.675,0.225,color,times,x,y);

    square(-0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
    square(0.675,0.45,color,times,x,y);

    square(0.45,0.675,color,times,x,y);
    square(0.675,0.675,color,times,x,y);
  }break;
  case 3:{ //Reverse New shape Left
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.9,-0.45,color,times,x,y);

    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.9,-0.225,color,times,x,y);

    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.9,0.0,color,times,x,y);

    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.9,0.225,color,times,x,y);

    square(0.0,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);
    square(-0.675,0.45,color,times,x,y);
    square(-0.9,0.45,color,times,x,y);

    square(-0.675,0.675,color,times,x,y);
    square(-0.9,0.675,color,times,x,y);
  }break;
  }
}

void draw_L1(int type, float x, float y, float color, float times) //L shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      square(-0.9,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(0.225,0.675,color,times,x,y);

      square(-0.9,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    square(0.675,0.675,color,times,x,y);
    square(0.45,0.675,color,times,x,y);
    square(0.225,0.675,color,times,x,y);
    square(0.0,0.675,color,times,x,y);
    square(-0.225,0.675,color,times,x,y);
    square(-0.45,0.675,color,times,x,y);

    square(0.675,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);
    square(0.0,0.45,color,times,x,y);
    square(-0.225,0.45,color,times,x,y);
    square(-0.45,0.45,color,times,x,y);

    square(0.675,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);

    square(0.675,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);

    square(0.675,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);

    square(0.675,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
  }break;
  case 2: // Reverse L Shape Left
    {
      square(-0.9,-0.9,color,times,x,y);
      square(-0.675,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);

      square(-0.9,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
    }break;
  case 3: // Reverse L shape Right
    {
      square(0.675,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);

      square(0.675,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);

      square(0.675,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);

      square(0.675,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);

      square(0.675,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);

      square(0.675,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
    }break;
  }
}

void draw_L2(int type, float x, float y, float color, float times) //L shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(0.225,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    square(0.225,0.225,color,times,x,y);
    square(0.0,0.225,color,times,x,y);
    square(-0.225,0.225,color,times,x,y);
    square(-0.45,0.225,color,times,x,y);
    square(-0.675,0.225,color,times,x,y);
    square(-0.9,0.225,color,times,x,y);

    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);
    square(-0.675,0.0,color,times,x,y);
    square(-0.9,0.0,color,times,x,y);

    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);
    square(-0.675,-0.225,color,times,x,y);
    square(-0.9,-0.225,color,times,x,y);

    square(-0.45,-0.45,color,times,x,y);
    square(-0.675,-0.45,color,times,x,y);
    square(-0.9,-0.45,color,times,x,y);

    square(-0.45,-0.675,color,times,x,y);
    square(-0.675,-0.675,color,times,x,y);
    square(-0.9,-0.675,color,times,x,y);

    square(-0.45,-0.9,color,times,x,y);
    square(-0.675,-0.9,color,times,x,y);
    square(-0.9,-0.9,color,times,x,y);
  }break;
  case 2: // Reverse L Shape Left
    {
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(0.225,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);
      square(0.675,0.45,color,times,x,y);

      square(0.225,0.675,color,times,x,y);
      square(0.45,0.675,color,times,x,y);
      square(0.675,0.675,color,times,x,y);
    }break;
  case 3: // Reverse L shape Right
    {
      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(-0.45,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.9,0.45,color,times,x,y);

      square(-0.45,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.9,0.675,color,times,x,y);
    }break;
  }
}

void draw_L3(int type, float x, float y, float color, float times) //L shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      square(-0.9,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);

      square(-0.9,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    square(0.675,0.675,color,times,x,y);
    square(0.45,0.675,color,times,x,y);
    square(0.225,0.675,color,times,x,y);

    square(0.675,0.45,color,times,x,y);
    square(0.45,0.45,color,times,x,y);
    square(0.225,0.45,color,times,x,y);

    square(0.675,0.225,color,times,x,y);
    square(0.45,0.225,color,times,x,y);
    square(0.225,0.225,color,times,x,y);

    square(0.675,0.0,color,times,x,y);
    square(0.45,0.0,color,times,x,y);
    square(0.225,0.0,color,times,x,y);
    square(0.0,0.0,color,times,x,y);
    square(-0.225,0.0,color,times,x,y);
    square(-0.45,0.0,color,times,x,y);

    square(0.675,-0.225,color,times,x,y);
    square(0.45,-0.225,color,times,x,y);
    square(0.225,-0.225,color,times,x,y);
    square(0.0,-0.225,color,times,x,y);
    square(-0.225,-0.225,color,times,x,y);
    square(-0.45,-0.225,color,times,x,y);

    square(0.675,-0.45,color,times,x,y);
    square(0.45,-0.45,color,times,x,y);
    square(0.225,-0.45,color,times,x,y);
    square(0.0,-0.45,color,times,x,y);
    square(-0.225,-0.45,color,times,x,y);
    square(-0.45,-0.45,color,times,x,y);
  }break;
  case 2: // Reverse L Shape Left
    {
      square(-0.9,-0.9,color,times,x,y);
      square(-0.675,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);

      square(-0.9,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);

      square(-0.9,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);

      square(-0.9,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);

      square(-0.9,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);

      square(-0.9,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
    }break;
  case 3: // Reverse L shape Right
    {
      square(0.675,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);

      square(0.675,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);

      square(0.675,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);

      square(0.675,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);

      square(0.675,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);

      square(0.675,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
    }break;
  }
}

/*
void draw_dist(int type, float x, float y, float color, float times)
{
  int anti_color;

  if(color == 1.0)anti_color = 0.0;
  else anti_color = 1.0;
  switch(type){
  case 0:
    {
      square(-0.45,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);

      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-1.125,-0.225,anti_color,times,x,y);
      square(-0.9,-0.225,anti_color,times,x,y);
      square(-0.675,-0.225,anti_color,times,x,y);

      square(-1.125,0.0,anti_color,times,x,y);
      square(-0.9,0.0,anti_color,times,x,y);
      square(-0.675,0.0,anti_color,times,x,y);

      square(-1.125,0.225,anti_color,times,x,y);
      square(-0.9,0.225,anti_color,times,x,y);
      square(-0.675,0.225,anti_color,times,x,y);

      square(-1.125,0.45,anti_color,times,x,y);
      square(-0.9,0.45,anti_color,times,x,y);
      square(-0.675,0.45,anti_color,times,x,y);
      square(-0.45,0.45,anti_color,times,x,y);
      square(-0.225,0.45,anti_color,times,x,y);
      square(0.0,0.45,anti_color,times,x,y);

      square(-1.125,0.675,anti_color,times,x,y);
      square(-0.9,0.675,anti_color,times,x,y);
      square(-0.675,0.675,anti_color,times,x,y);
      square(-0.45,0.675,anti_color,times,x,y);
      square(-0.225,0.675,anti_color,times,x,y);
      square(0.0,0.675,anti_color,times,x,y);

      square(-1.125,0.9,anti_color,times,x,y);
      square(-0.9,0.9,anti_color,times,x,y);
      square(-0.675,0.9,anti_color,times,x,y);
      square(-0.45,0.9,anti_color,times,x,y);
      square(-0.225,0.9,anti_color,times,x,y);
      square(0.0,0.9,anti_color,times,x,y);
    }break;
  case 1:
    {
      square(0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);
      square(-0.675,-0.9,color,times,x,y);
      square(-0.9,-0.9,color,times,x,y);

      square(0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.9,-0.675,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(0.9,-0.225,anti_color,times,x,y);
      square(0.675,-0.225,anti_color,times,x,y);
      square(0.45,-0.225,anti_color,times,x,y);

      square(0.9,0.0,anti_color,times,x,y);
      square(0.675,0.0,anti_color,times,x,y);
      square(0.45,0.0,anti_color,times,x,y);

      square(0.9,0.225,anti_color,times,x,y);
      square(0.675,0.225,anti_color,times,x,y);
      square(0.45,0.225,anti_color,times,x,y);

      square(0.9,0.45,anti_color,times,x,y);
      square(0.675,0.45,anti_color,times,x,y);
      square(0.45,0.45,anti_color,times,x,y);
      square(0.225,0.45,anti_color,times,x,y);
      square(0.0,0.45,anti_color,times,x,y);
      square(-0.225,0.45,anti_color,times,x,y);

      square(0.9,0.675,anti_color,times,x,y);
      square(0.675,0.675,anti_color,times,x,y);
      square(0.45,0.675,anti_color,times,x,y);
      square(0.225,0.675,anti_color,times,x,y);
      square(0.0,0.675,anti_color,times,x,y);
      square(-0.225,0.675,anti_color,times,x,y);

      square(0.9,0.9,anti_color,times,x,y);
      square(0.675,0.9,anti_color,times,x,y);
      square(0.45,0.9,anti_color,times,x,y);
      square(0.225,0.9,anti_color,times,x,y);
      square(0.0,0.9,anti_color,times,x,y);
      square(-0.225,0.9,anti_color,times,x,y);
    }break;
  case 2:
    {
      square(-0.45,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(0.225,0.675,color,times,x,y);
      square(0.45,0.675,color,times,x,y);
      square(0.675,0.675,color,times,x,y);

      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);
      square(0.675,0.45,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-1.125,0.0,anti_color,times,x,y);
      square(-0.9,0.0,anti_color,times,x,y);
      square(-0.675,0.0,anti_color,times,x,y);

      square(-1.125,-0.225,anti_color,times,x,y);
      square(-0.9,-0.225,anti_color,times,x,y);
      square(-0.675,-0.225,anti_color,times,x,y);

      square(-1.125,-0.45,anti_color,times,x,y);
      square(-0.9,-0.45,anti_color,times,x,y);
      square(-0.675,-0.45,anti_color,times,x,y);

      square(-1.125,-0.675,anti_color,times,x,y);
      square(-0.9,-0.675,anti_color,times,x,y);
      square(-0.675,-0.675,anti_color,times,x,y);
      square(-0.45,-0.675,anti_color,times,x,y);
      square(-0.225,-0.675,anti_color,times,x,y);
      square(0.0,-0.675,anti_color,times,x,y);

      square(-1.125,-0.9,anti_color,times,x,y);
      square(-0.9,-0.9,anti_color,times,x,y);
      square(-0.675,-0.9,anti_color,times,x,y);
      square(-0.45,-0.9,anti_color,times,x,y);
      square(-0.225,-0.9,anti_color,times,x,y);
      square(0.0,-0.9,anti_color,times,x,y);

      square(-1.125,-1.125,anti_color,times,x,y);
      square(-0.9,-1.125,anti_color,times,x,y);
      square(-0.675,-1.125,anti_color,times,x,y);
      square(-0.45,-1.125,anti_color,times,x,y);
      square(-0.225,-1.125,anti_color,times,x,y);
      square(0.0,-1.125,anti_color,times,x,y);
    }break;
  case 3:
    {
      square(0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.9,0.675,color,times,x,y);

      square(0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.9,0.45,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.9,0.0,anti_color,times,x,y);
      square(0.675,0.0,anti_color,times,x,y);
      square(0.45,0.0,anti_color,times,x,y);

      square(0.9,-0.225,anti_color,times,x,y);
      square(0.675,-0.225,anti_color,times,x,y);
      square(0.45,-0.225,anti_color,times,x,y);

      square(0.9,-0.45,anti_color,times,x,y);
      square(0.675,-0.45,anti_color,times,x,y);
      square(0.45,-0.45,anti_color,times,x,y);

      square(0.9,-0.675,anti_color,times,x,y);
      square(0.675,-0.675,anti_color,times,x,y);
      square(0.45,-0.675,anti_color,times,x,y);
      square(0.225,-0.675,anti_color,times,x,y);
      square(0.0,-0.675,anti_color,times,x,y);
      square(-0.225,-0.675,anti_color,times,x,y);

      square(0.9,-0.9,anti_color,times,x,y);
      square(0.675,-0.9,anti_color,times,x,y);
      square(0.45,-0.9,anti_color,times,x,y);
      square(0.225,-0.9,anti_color,times,x,y);
      square(0.0,-0.9,anti_color,times,x,y);
      square(-0.225,-0.9,anti_color,times,x,y);

      square(0.9,-1.125,anti_color,times,x,y);
      square(0.675,-1.125,anti_color,times,x,y);
      square(0.45,-1.125,anti_color,times,x,y);
      square(0.225,-1.125,anti_color,times,x,y);
      square(0.0,-1.125,anti_color,times,x,y);
      square(-0.225,-1.125,anti_color,times,x,y);
    }break;
  }
}
*/

void draw_dist(int type, float x, float y, float color, float times)
{
  float anti_color;

  if(color == 1.0)anti_color = 0.0;
  else anti_color = 1.0;
  switch(type){
  case 0:
    {
      /*square(-0.45,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);

      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);*/
      /******************************************/
      square(-1.125,-0.225,anti_color,times,x,y);
      square(-0.9,-0.225,anti_color,times,x,y);
      square(-0.675,-0.225,anti_color,times,x,y);

      square(-1.125,0.0,anti_color,times,x,y);
      square(-0.9,0.0,anti_color,times,x,y);
      square(-0.675,0.0,anti_color,times,x,y);

      square(-1.125,0.225,anti_color,times,x,y);
      square(-0.9,0.225,anti_color,times,x,y);
      square(-0.675,0.225,anti_color,times,x,y);

      square(-1.125,0.45,anti_color,times,x,y);
      square(-0.9,0.45,anti_color,times,x,y);
      square(-0.675,0.45,anti_color,times,x,y);
      square(-0.45,0.45,anti_color,times,x,y);
      square(-0.225,0.45,anti_color,times,x,y);
      square(0.0,0.45,anti_color,times,x,y);

      square(-1.125,0.675,anti_color,times,x,y);
      square(-0.9,0.675,anti_color,times,x,y);
      square(-0.675,0.675,anti_color,times,x,y);
      square(-0.45,0.675,anti_color,times,x,y);
      square(-0.225,0.675,anti_color,times,x,y);
      square(0.0,0.675,anti_color,times,x,y);

      square(-1.125,0.9,anti_color,times,x,y);
      square(-0.9,0.9,anti_color,times,x,y);
      square(-0.675,0.9,anti_color,times,x,y);
      square(-0.45,0.9,anti_color,times,x,y);
      square(-0.225,0.9,anti_color,times,x,y);
      square(0.0,0.9,anti_color,times,x,y);

      square(-0.45,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(0.225,-0.9,color,times,x,y);
      square(0.45,-0.9,color,times,x,y);
      square(0.675,-0.9,color,times,x,y);

      square(-0.45,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(0.225,-0.675,color,times,x,y);
      square(0.45,-0.675,color,times,x,y);
      square(0.675,-0.675,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

    }break;
  case 1:
    {
      /*square(0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);
      square(-0.675,-0.9,color,times,x,y);
      square(-0.9,-0.9,color,times,x,y);

      square(0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.9,-0.675,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);*/
      /************************************/
      square(0.9,-0.225,anti_color,times,x,y);
      square(0.675,-0.225,anti_color,times,x,y);
      square(0.45,-0.225,anti_color,times,x,y);

      square(0.9,0.0,anti_color,times,x,y);
      square(0.675,0.0,anti_color,times,x,y);
      square(0.45,0.0,anti_color,times,x,y);

      square(0.9,0.225,anti_color,times,x,y);
      square(0.675,0.225,anti_color,times,x,y);
      square(0.45,0.225,anti_color,times,x,y);

      square(0.9,0.45,anti_color,times,x,y);
      square(0.675,0.45,anti_color,times,x,y);
      square(0.45,0.45,anti_color,times,x,y);
      square(0.225,0.45,anti_color,times,x,y);
      square(0.0,0.45,anti_color,times,x,y);
      square(-0.225,0.45,anti_color,times,x,y);

      square(0.9,0.675,anti_color,times,x,y);
      square(0.675,0.675,anti_color,times,x,y);
      square(0.45,0.675,anti_color,times,x,y);
      square(0.225,0.675,anti_color,times,x,y);
      square(0.0,0.675,anti_color,times,x,y);
      square(-0.225,0.675,anti_color,times,x,y);

      square(0.9,0.9,anti_color,times,x,y);
      square(0.675,0.9,anti_color,times,x,y);
      square(0.45,0.9,anti_color,times,x,y);
      square(0.225,0.9,anti_color,times,x,y);
      square(0.0,0.9,anti_color,times,x,y);
      square(-0.225,0.9,anti_color,times,x,y);

      square(0.225,-0.9,color,times,x,y);
      square(0.0,-0.9,color,times,x,y);
      square(-0.225,-0.9,color,times,x,y);
      square(-0.45,-0.9,color,times,x,y);
      square(-0.675,-0.9,color,times,x,y);
      square(-0.9,-0.9,color,times,x,y);

      square(0.225,-0.675,color,times,x,y);
      square(0.0,-0.675,color,times,x,y);
      square(-0.225,-0.675,color,times,x,y);
      square(-0.45,-0.675,color,times,x,y);
      square(-0.675,-0.675,color,times,x,y);
      square(-0.9,-0.675,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);
    }break;
  case 2:
    {
      /*square(-0.45,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(0.225,0.675,color,times,x,y);
      square(0.45,0.675,color,times,x,y);
      square(0.675,0.675,color,times,x,y);

      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);
      square(0.675,0.45,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);*/
      /****************************************/
      square(-1.125,0.0,anti_color,times,x,y);
      square(-0.9,0.0,anti_color,times,x,y);
      square(-0.675,0.0,anti_color,times,x,y);

      square(-1.125,-0.225,anti_color,times,x,y);
      square(-0.9,-0.225,anti_color,times,x,y);
      square(-0.675,-0.225,anti_color,times,x,y);

      square(-1.125,-0.45,anti_color,times,x,y);
      square(-0.9,-0.45,anti_color,times,x,y);
      square(-0.675,-0.45,anti_color,times,x,y);

      square(-1.125,-0.675,anti_color,times,x,y);
      square(-0.9,-0.675,anti_color,times,x,y);
      square(-0.675,-0.675,anti_color,times,x,y);
      square(-0.45,-0.675,anti_color,times,x,y);
      square(-0.225,-0.675,anti_color,times,x,y);
      square(0.0,-0.675,anti_color,times,x,y);

      square(-1.125,-0.9,anti_color,times,x,y);
      square(-0.9,-0.9,anti_color,times,x,y);
      square(-0.675,-0.9,anti_color,times,x,y);
      square(-0.45,-0.9,anti_color,times,x,y);
      square(-0.225,-0.9,anti_color,times,x,y);
      square(0.0,-0.9,anti_color,times,x,y);

      square(-1.125,-1.125,anti_color,times,x,y);
      square(-0.9,-1.125,anti_color,times,x,y);
      square(-0.675,-1.125,anti_color,times,x,y);
      square(-0.45,-1.125,anti_color,times,x,y);
      square(-0.225,-1.125,anti_color,times,x,y);
      square(0.0,-1.125,anti_color,times,x,y);

      square(-0.45,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(0.225,0.675,color,times,x,y);
      square(0.45,0.675,color,times,x,y);
      square(0.675,0.675,color,times,x,y);

      square(-0.45,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(0.225,0.45,color,times,x,y);
      square(0.45,0.45,color,times,x,y);
      square(0.675,0.45,color,times,x,y);

      square(-0.45,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(0.225,0.225,color,times,x,y);
      square(0.45,0.225,color,times,x,y);
      square(0.675,0.225,color,times,x,y);

      square(-0.45,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(0.225,0.0,color,times,x,y);
      square(0.45,0.0,color,times,x,y);
      square(0.675,0.0,color,times,x,y);

      square(-0.45,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(0.225,-0.225,color,times,x,y);
      square(0.45,-0.225,color,times,x,y);
      square(0.675,-0.225,color,times,x,y);

      square(-0.45,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(0.225,-0.45,color,times,x,y);
      square(0.45,-0.45,color,times,x,y);
      square(0.675,-0.45,color,times,x,y);
    }break;
  case 3:
    {
      /*square(0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.9,0.675,color,times,x,y);

      square(0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.9,0.45,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);*/
      /**************************************/
      square(0.9,0.0,anti_color,times,x,y);
      square(0.675,0.0,anti_color,times,x,y);
      square(0.45,0.0,anti_color,times,x,y);

      square(0.9,-0.225,anti_color,times,x,y);
      square(0.675,-0.225,anti_color,times,x,y);
      square(0.45,-0.225,anti_color,times,x,y);

      square(0.9,-0.45,anti_color,times,x,y);
      square(0.675,-0.45,anti_color,times,x,y);
      square(0.45,-0.45,anti_color,times,x,y);

      square(0.9,-0.675,anti_color,times,x,y);
      square(0.675,-0.675,anti_color,times,x,y);
      square(0.45,-0.675,anti_color,times,x,y);
      square(0.225,-0.675,anti_color,times,x,y);
      square(0.0,-0.675,anti_color,times,x,y);
      square(-0.225,-0.675,anti_color,times,x,y);

      square(0.9,-0.9,anti_color,times,x,y);
      square(0.675,-0.9,anti_color,times,x,y);
      square(0.45,-0.9,anti_color,times,x,y);
      square(0.225,-0.9,anti_color,times,x,y);
      square(0.0,-0.9,anti_color,times,x,y);
      square(-0.225,-0.9,anti_color,times,x,y);

      square(0.9,-1.125,anti_color,times,x,y);
      square(0.675,-1.125,anti_color,times,x,y);
      square(0.45,-1.125,anti_color,times,x,y);
      square(0.225,-1.125,anti_color,times,x,y);
      square(0.0,-1.125,anti_color,times,x,y);
      square(-0.225,-1.125,anti_color,times,x,y);

      square(0.225,0.675,color,times,x,y);
      square(0.0,0.675,color,times,x,y);
      square(-0.225,0.675,color,times,x,y);
      square(-0.45,0.675,color,times,x,y);
      square(-0.675,0.675,color,times,x,y);
      square(-0.9,0.675,color,times,x,y);

      square(0.225,0.45,color,times,x,y);
      square(0.0,0.45,color,times,x,y);
      square(-0.225,0.45,color,times,x,y);
      square(-0.45,0.45,color,times,x,y);
      square(-0.675,0.45,color,times,x,y);
      square(-0.9,0.45,color,times,x,y);

      square(0.225,0.225,color,times,x,y);
      square(0.0,0.225,color,times,x,y);
      square(-0.225,0.225,color,times,x,y);
      square(-0.45,0.225,color,times,x,y);
      square(-0.675,0.225,color,times,x,y);
      square(-0.9,0.225,color,times,x,y);

      square(0.225,0.0,color,times,x,y);
      square(0.0,0.0,color,times,x,y);
      square(-0.225,0.0,color,times,x,y);
      square(-0.45,0.0,color,times,x,y);
      square(-0.675,0.0,color,times,x,y);
      square(-0.9,0.0,color,times,x,y);

      square(0.225,-0.225,color,times,x,y);
      square(0.0,-0.225,color,times,x,y);
      square(-0.225,-0.225,color,times,x,y);
      square(-0.45,-0.225,color,times,x,y);
      square(-0.675,-0.225,color,times,x,y);
      square(-0.9,-0.225,color,times,x,y);

      square(0.225,-0.45,color,times,x,y);
      square(0.0,-0.45,color,times,x,y);
      square(-0.225,-0.45,color,times,x,y);
      square(-0.45,-0.45,color,times,x,y);
      square(-0.675,-0.45,color,times,x,y);
      square(-0.9,-0.45,color,times,x,y);
    }break;
  }
}

void shape_Sq(int type, float x, float y, float color, float times) //New Shape
{
  switch(type){
  case 0: //New shape is Right
    {
      edge_col(0.675,-0.675,color,times,x,y);
      edge_col(0.675,-0.45,color,times,x,y);
      edge_col(0.675,-0.225,color,times,x,y);
      edge_col(0.675,0.0,color,times,x,y);
      edge_col(0.675,0.225,color,times,x,y);
      edge_col(0.675,0.45,color,times,x,y);

      edge_col(-0.675,-0.675,color,times,x,y);
      edge_col(-0.675,-0.45,color,times,x,y);
      edge_col(-0.675,-0.225,color,times,x,y);
      edge_col(-0.675,0.0,color,times,x,y);
      edge_col(-0.675,0.225,color,times,x,y);
      edge_col(-0.675,0.45,color,times,x,y);

      edge_row(0.45,0.675,color,times,x,y);
      edge_row(0.225,0.675,color,times,x,y);
      edge_row(0.0,0.675,color,times,x,y);
      edge_row(-0.225,0.675,color,times,x,y);
      edge_row(-0.45,0.675,color,times,x,y);
      edge_row(-0.675,0.675,color,times,x,y);

      edge_row(0.45,-0.675,color,times,x,y);
      edge_row(0.225,-0.675,color,times,x,y);
      edge_row(0.0,-0.675,color,times,x,y);
      edge_row(-0.225,-0.675,color,times,x,y);
      edge_row(-0.45,-0.675,color,times,x,y);
      edge_row(-0.675,-0.675,color,times,x,y);
    }break;
  case 1:{ //Nwe shape is Left
    edge_col(-0.675,-0.675,color,times,x,y);
    edge_col(-0.675,-0.45,color,times,x,y);
    edge_col(-0.675,-0.225,color,times,x,y);
    edge_col(-0.675,0.0,color,times,x,y);
    edge_col(-0.675,0.225,color,times,x,y);
    edge_col(-0.675,0.45,color,times,x,y);

    edge_col(0.675,-0.675,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,0.0,color,times,x,y);
    edge_col(0.675,0.225,color,times,x,y);
    edge_col(0.675,0.45,color,times,x,y);

    edge_row(-0.675,0.675,color,times,x,y);
    edge_row(-0.45,0.675,color,times,x,y);
    edge_row(-0.225,0.675,color,times,x,y);
    edge_row(0.0,0.675,color,times,x,y);
    edge_row(0.225,0.675,color,times,x,y);
    edge_row(0.45,0.675,color,times,x,y);

    edge_row(-0.675,-0.675,color,times,x,y);
    edge_row(-0.45,-0.675,color,times,x,y);
    edge_row(-0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.45,-0.675,color,times,x,y);
  }break;
  case 2:{
    edge_col(0.675,0.45,color,times,x,y);
    edge_col(0.675,0.225,color,times,x,y);
    edge_col(0.675,0.0,color,times,x,y);
    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    edge_col(-0.675,0.45,color,times,x,y);
    edge_col(-0.675,0.225,color,times,x,y);
    edge_col(-0.675,0.0,color,times,x,y);
    edge_col(-0.675,-0.225,color,times,x,y);
    edge_col(-0.675,-0.45,color,times,x,y);
    edge_col(-0.675,-0.675,color,times,x,y);

    edge_row(0.45,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);
    edge_row(-0.225,-0.675,color,times,x,y);
    edge_row(-0.45,-0.675,color,times,x,y);
    edge_row(-0.675,-0.675,color,times,x,y);

    edge_row(0.45,0.675,color,times,x,y);
    edge_row(0.225,0.675,color,times,x,y);
    edge_row(0.0,0.675,color,times,x,y);
    edge_row(-0.225,0.675,color,times,x,y);
    edge_row(-0.45,0.675,color,times,x,y);
    edge_row(-0.675,0.675,color,times,x,y);
  }break;
  case 3:{
    edge_col(-0.675,0.45,color,times,x,y);
    edge_col(-0.675,0.225,color,times,x,y);
    edge_col(-0.675,0.0,color,times,x,y);
    edge_col(-0.675,-0.225,color,times,x,y);
    edge_col(-0.675,-0.45,color,times,x,y);
    edge_col(-0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    edge_col(0.675,0.45,color,times,x,y);
    edge_col(0.675,0.225,color,times,x,y);
    edge_col(0.675,0.0,color,times,x,y);
    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.675,color,times,x,y);

    edge_row(-0.675,-0.675,color,times,x,y);
    edge_row(-0.45,-0.675,color,times,x,y);
    edge_row(-0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.45,-0.675,color,times,x,y);

    edge_row(-0.675,0.675,color,times,x,y);
    edge_row(-0.45,0.675,color,times,x,y);
    edge_row(-0.225,0.675,color,times,x,y);
    edge_row(0.0,0.675,color,times,x,y);
    edge_row(0.225,0.675,color,times,x,y);
    edge_row(0.45,0.675,color,times,x,y);
  }break;
  }
}

void shape_New(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      edge_row(0.225,0.675,color,times,x,y);
      edge_row(0.45,0.675,color,times,x,y);

      edge_row(-0.45,0.45,color,times,x,y);
      edge_row(-0.225,0.45,color,times,x,y);
      edge_row(0.0,0.45,color,times,x,y);

      edge_row(-0.675,-0.225,color,times,x,y);

      edge_row(-0.675,-0.675,color,times,x,y);
      edge_row(-0.45,-0.675,color,times,x,y);
      edge_row(-0.225,-0.675,color,times,x,y);
      edge_row(0.0,-0.675,color,times,x,y);
      edge_row(0.225,-0.675,color,times,x,y);
      edge_row(0.45,-0.675,color,times,x,y);

      edge_col(-0.675,-0.675,color,times,x,y);
      edge_col(-0.675,-0.45,color,times,x,y);

      edge_col(-0.45,-0.225,color,times,x,y);
      edge_col(-0.45,0.0,color,times,x,y);
      edge_col(-0.45,0.225,color,times,x,y);

      edge_col(0.225,0.45,color,times,x,y);

      edge_col(0.675,-0.675,color,times,x,y);
      edge_col(0.675,-0.45,color,times,x,y);
      edge_col(0.675,-0.225,color,times,x,y);
      edge_col(0.675,0.0,color,times,x,y);
      edge_col(0.675,0.225,color,times,x,y);
      edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:
    {
      edge_row(-0.45,0.675,color,times,x,y);
      edge_row(-0.675,0.675,color,times,x,y);

      edge_row(0.225,0.45,color,times,x,y);
      edge_row(0.0,0.45,color,times,x,y);
      edge_row(-0.225,0.45,color,times,x,y);

      edge_row(0.45,-0.225,color,times,x,y);

      edge_row(0.45,-0.675,color,times,x,y);
      edge_row(0.225,-0.675,color,times,x,y);
      edge_row(0.0,-0.675,color,times,x,y);
      edge_row(-0.225,-0.675,color,times,x,y);
      edge_row(-0.45,-0.675,color,times,x,y);
      edge_row(-0.675,-0.675,color,times,x,y);

      edge_col(0.675,-0.675,color,times,x,y);
      edge_col(0.675,-0.45,color,times,x,y);

      edge_col(0.45,-0.225,color,times,x,y);
      edge_col(0.45,0.0,color,times,x,y);
      edge_col(0.45,0.225,color,times,x,y);

      edge_col(-0.225,0.45,color,times,x,y);

      edge_col(-0.675,-0.675,color,times,x,y);
      edge_col(-0.675,-0.45,color,times,x,y);
      edge_col(-0.675,-0.225,color,times,x,y);
      edge_col(-0.675,0.0,color,times,x,y);
      edge_col(-0.675,0.225,color,times,x,y);
      edge_col(-0.675,0.45,color,times,x,y);
    }break;
  case 2:
    {
      edge_row(0.225,-0.675,color,times,x,y);
      edge_row(0.45,-0.675,color,times,x,y);

      edge_row(-0.45,-0.45,color,times,x,y);
      edge_row(-0.225,-0.45,color,times,x,y);
      edge_row(0.0,-0.45,color,times,x,y);

      edge_row(-0.675,0.225,color,times,x,y);

      edge_row(-0.675,0.675,color,times,x,y);
      edge_row(-0.45,0.675,color,times,x,y);
      edge_row(-0.225,0.675,color,times,x,y);
      edge_row(0.0,0.675,color,times,x,y);
      edge_row(0.225,0.675,color,times,x,y);
      edge_row(0.45,0.675,color,times,x,y);

      edge_col(-0.675,0.45,color,times,x,y);
      edge_col(-0.675,0.225,color,times,x,y);

      edge_col(-0.45,0.0,color,times,x,y);
      edge_col(-0.45,-0.225,color,times,x,y);
      edge_col(-0.45,-0.45,color,times,x,y);

      edge_col(0.225,-0.675,color,times,x,y);

      edge_col(0.675,0.45,color,times,x,y);
      edge_col(0.675,0.225,color,times,x,y);
      edge_col(0.675,0.0,color,times,x,y);
      edge_col(0.675,-0.225,color,times,x,y);
      edge_col(0.675,-0.45,color,times,x,y);
      edge_col(0.675,-0.675,color,times,x,y);
    }break;
  case 3:
    {
      edge_row(-0.45,-0.675,color,times,x,y);
      edge_row(-0.675,-0.675,color,times,x,y);

      edge_row(0.225,-0.45,color,times,x,y);
      edge_row(0.0,-0.45,color,times,x,y);
      edge_row(-0.225,-0.45,color,times,x,y);

      edge_row(0.45,0.225,color,times,x,y);

      edge_row(0.45,0.675,color,times,x,y);
      edge_row(0.225,0.675,color,times,x,y);
      edge_row(0.0,0.675,color,times,x,y);
      edge_row(-0.225,0.675,color,times,x,y);
      edge_row(-0.45,0.675,color,times,x,y);
      edge_row(-0.675,0.675,color,times,x,y);

      edge_col(0.675,0.45,color,times,x,y);
      edge_col(0.675,0.225,color,times,x,y);

      edge_col(0.45,0.0,color,times,x,y);
      edge_col(0.45,-0.225,color,times,x,y);
      edge_col(0.45,-0.45,color,times,x,y);

      edge_col(-0.225,-0.675,color,times,x,y);

      edge_col(-0.675,0.45,color,times,x,y);
      edge_col(-0.675,0.225,color,times,x,y);
      edge_col(-0.675,0.0,color,times,x,y);
      edge_col(-0.675,-0.225,color,times,x,y);
      edge_col(-0.675,-0.45,color,times,x,y);
      edge_col(-0.675,-0.675,color,times,x,y);
    }break;
  }
}

void shape_L(int type, float x, float y, float color, float times) //L Shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      edge_row(-0.675,0.675,color,times,x,y);
      edge_row(-0.45,0.675,color,times,x,y);
      edge_row(-0.225,0.675,color,times,x,y);
      edge_row(0.0,0.675,color,times,x,y);
      edge_row(0.225,0.675,color,times,x,y);
      edge_row(0.45,0.675,color,times,x,y);

      edge_row(0.0,0.0,color,times,x,y);
      edge_row(0.225,0.0,color,times,x,y);
      edge_row(0.45,0.0,color,times,x,y);

      edge_row(-0.675,-0.675,color,times,x,y);
      edge_row(-0.45,-0.675,color,times,x,y);
      edge_row(-0.225,-0.675,color,times,x,y);

      edge_col(-0.675,-0.675,color,times,x,y);
      edge_col(-0.675,-0.45,color,times,x,y);
      edge_col(-0.675,-0.225,color,times,x,y);
      edge_col(-0.675,0.0,color,times,x,y);
      edge_col(-0.675,0.225,color,times,x,y);
      edge_col(-0.675,0.45,color,times,x,y);

      edge_col(0.0,-0.675,color,times,x,y);
      edge_col(0.0,-0.45,color,times,x,y);
      edge_col(0.0,-0.225,color,times,x,y);

      edge_col(0.675,0.0,color,times,x,y);
      edge_col(0.675,0.225,color,times,x,y);
      edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    edge_row(0.45,0.675,color,times,x,y);
    edge_row(0.225,0.675,color,times,x,y);
    edge_row(0.0,0.675,color,times,x,y);
    edge_row(-0.225,0.675,color,times,x,y);
    edge_row(-0.45,0.675,color,times,x,y);
    edge_row(-0.675,0.675,color,times,x,y);

    edge_row(-0.225,0.0,color,times,x,y);
    edge_row(-0.45,0.0,color,times,x,y);
    edge_row(-0.675,0.0,color,times,x,y);

    edge_row(0.45,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);

    edge_col(0.675,-0.675,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,0.0,color,times,x,y);
    edge_col(0.675,0.225,color,times,x,y);
    edge_col(0.675,0.45,color,times,x,y);

    edge_col(0.0,-0.675,color,times,x,y);
    edge_col(0.0,-0.45,color,times,x,y);
    edge_col(0.0,-0.225,color,times,x,y);

    edge_col(-0.675,0.0,color,times,x,y);
    edge_col(-0.675,0.225,color,times,x,y);
    edge_col(-0.675,0.45,color,times,x,y);
  }break;
  case 2:{
    edge_row(-0.675,-0.675,color,times,x,y);
    edge_row(-0.45,-0.675,color,times,x,y);
    edge_row(-0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.45,-0.675,color,times,x,y);

    edge_row(0.0,0.0,color,times,x,y);
    edge_row(0.225,0.0,color,times,x,y);
    edge_row(0.45,0.0,color,times,x,y);

    edge_row(-0.675,0.675,color,times,x,y);
    edge_row(-0.45,0.675,color,times,x,y);
    edge_row(-0.225,0.675,color,times,x,y);

    edge_col(-0.675,0.45,color,times,x,y);
    edge_col(-0.675,0.225,color,times,x,y);
    edge_col(-0.675,0.0,color,times,x,y);
    edge_col(-0.675,-0.225,color,times,x,y);
    edge_col(-0.675,-0.45,color,times,x,y);
    edge_col(-0.675,-0.675,color,times,x,y);

    edge_col(0.0,0.45,color,times,x,y);
    edge_col(0.0,0.225,color,times,x,y);
    edge_col(0.0,0.0,color,times,x,y);

    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.675,color,times,x,y);
  }break;
  case 3:{
    edge_row(0.45,-0.675,color,times,x,y);
    edge_row(0.225,-0.675,color,times,x,y);
    edge_row(0.0,-0.675,color,times,x,y);
    edge_row(-0.225,-0.675,color,times,x,y);
    edge_row(-0.45,-0.675,color,times,x,y);
    edge_row(-0.675,-0.675,color,times,x,y);

    edge_row(-0.225,0.0,color,times,x,y);
    edge_row(-0.45,0.0,color,times,x,y);
    edge_row(-0.675,0.0,color,times,x,y);

    edge_row(0.45,0.675,color,times,x,y);
    edge_row(0.225,0.675,color,times,x,y);
    edge_row(0.0,0.675,color,times,x,y);

    edge_col(0.675,0.45,color,times,x,y);
    edge_col(0.675,0.225,color,times,x,y);
    edge_col(0.675,0.0,color,times,x,y);
    edge_col(0.675,-0.225,color,times,x,y);
    edge_col(0.675,-0.45,color,times,x,y);
    edge_col(0.675,-0.675,color,times,x,y);

    edge_col(0.0,0.45,color,times,x,y);
    edge_col(0.0,0.225,color,times,x,y);
    edge_col(0.0,0.0,color,times,x,y);

    edge_col(-0.675,-0.225,color,times,x,y);
    edge_col(-0.675,-0.45,color,times,x,y);
    edge_col(-0.675,-0.675,color,times,x,y);
  }break;
  }
}

void blue_Sq(int type, float x, float y, float color, float times) //Square Shape
{
  switch(type){
  case 0: //New shape is Right
    {
      blue_edge_col(0.675,-0.675,color,times,x,y);
      blue_edge_col(0.675,-0.45,color,times,x,y);
      blue_edge_col(0.675,-0.225,color,times,x,y);
      blue_edge_col(0.675,0.0,color,times,x,y);
      blue_edge_col(0.675,0.225,color,times,x,y);
      blue_edge_col(0.675,0.45,color,times,x,y);

      blue_edge_col(-0.675,-0.675,color,times,x,y);
      blue_edge_col(-0.675,-0.45,color,times,x,y);
      blue_edge_col(-0.675,-0.225,color,times,x,y);
      blue_edge_col(-0.675,0.0,color,times,x,y);
      blue_edge_col(-0.675,0.225,color,times,x,y);
      blue_edge_col(-0.675,0.45,color,times,x,y);

      blue_edge_row(0.45,0.675,color,times,x,y);
      blue_edge_row(0.225,0.675,color,times,x,y);
      blue_edge_row(0.0,0.675,color,times,x,y);
      blue_edge_row(-0.225,0.675,color,times,x,y);
      blue_edge_row(-0.45,0.675,color,times,x,y);
      blue_edge_row(-0.675,0.675,color,times,x,y);

      blue_edge_row(0.45,-0.675,color,times,x,y);
      blue_edge_row(0.225,-0.675,color,times,x,y);
      blue_edge_row(0.0,-0.675,color,times,x,y);
      blue_edge_row(-0.225,-0.675,color,times,x,y);
      blue_edge_row(-0.45,-0.675,color,times,x,y);
      blue_edge_row(-0.675,-0.675,color,times,x,y);
    }break;
  case 1:{ //Nwe shape is Left
    blue_edge_col(-0.675,-0.675,color,times,x,y);
    blue_edge_col(-0.675,-0.45,color,times,x,y);
    blue_edge_col(-0.675,-0.225,color,times,x,y);
    blue_edge_col(-0.675,0.0,color,times,x,y);
    blue_edge_col(-0.675,0.225,color,times,x,y);
    blue_edge_col(-0.675,0.45,color,times,x,y);

    blue_edge_col(0.675,-0.675,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,0.0,color,times,x,y);
    blue_edge_col(0.675,0.225,color,times,x,y);
    blue_edge_col(0.675,0.45,color,times,x,y);

    blue_edge_row(-0.675,0.675,color,times,x,y);
    blue_edge_row(-0.45,0.675,color,times,x,y);
    blue_edge_row(-0.225,0.675,color,times,x,y);
    blue_edge_row(0.0,0.675,color,times,x,y);
    blue_edge_row(0.225,0.675,color,times,x,y);
    blue_edge_row(0.45,0.675,color,times,x,y);

    blue_edge_row(-0.675,-0.675,color,times,x,y);
    blue_edge_row(-0.45,-0.675,color,times,x,y);
    blue_edge_row(-0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.45,-0.675,color,times,x,y);
  }break;
  case 2:{
    blue_edge_col(0.675,0.45,color,times,x,y);
    blue_edge_col(0.675,0.225,color,times,x,y);
    blue_edge_col(0.675,0.0,color,times,x,y);
    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    blue_edge_col(-0.675,0.45,color,times,x,y);
    blue_edge_col(-0.675,0.225,color,times,x,y);
    blue_edge_col(-0.675,0.0,color,times,x,y);
    blue_edge_col(-0.675,-0.225,color,times,x,y);
    blue_edge_col(-0.675,-0.45,color,times,x,y);
    blue_edge_col(-0.675,-0.675,color,times,x,y);

    blue_edge_row(0.45,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);
    blue_edge_row(-0.225,-0.675,color,times,x,y);
    blue_edge_row(-0.45,-0.675,color,times,x,y);
    blue_edge_row(-0.675,-0.675,color,times,x,y);

    blue_edge_row(0.45,0.675,color,times,x,y);
    blue_edge_row(0.225,0.675,color,times,x,y);
    blue_edge_row(0.0,0.675,color,times,x,y);
    blue_edge_row(-0.225,0.675,color,times,x,y);
    blue_edge_row(-0.45,0.675,color,times,x,y);
    blue_edge_row(-0.675,0.675,color,times,x,y);
  }break;
  case 3:{
    blue_edge_col(-0.675,0.45,color,times,x,y);
    blue_edge_col(-0.675,0.225,color,times,x,y);
    blue_edge_col(-0.675,0.0,color,times,x,y);
    blue_edge_col(-0.675,-0.225,color,times,x,y);
    blue_edge_col(-0.675,-0.45,color,times,x,y);
    blue_edge_col(-0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    blue_edge_col(0.675,0.45,color,times,x,y);
    blue_edge_col(0.675,0.225,color,times,x,y);
    blue_edge_col(0.675,0.0,color,times,x,y);
    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.675,color,times,x,y);

    blue_edge_row(-0.675,-0.675,color,times,x,y);
    blue_edge_row(-0.45,-0.675,color,times,x,y);
    blue_edge_row(-0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.45,-0.675,color,times,x,y);

    blue_edge_row(-0.675,0.675,color,times,x,y);
    blue_edge_row(-0.45,0.675,color,times,x,y);
    blue_edge_row(-0.225,0.675,color,times,x,y);
    blue_edge_row(0.0,0.675,color,times,x,y);
    blue_edge_row(0.225,0.675,color,times,x,y);
    blue_edge_row(0.45,0.675,color,times,x,y);
  }break;
  }
}

void blue_New(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      blue_edge_row(0.225,0.675,color,times,x,y);
      blue_edge_row(0.45,0.675,color,times,x,y);

      blue_edge_row(-0.45,0.45,color,times,x,y);
      blue_edge_row(-0.225,0.45,color,times,x,y);
      blue_edge_row(0.0,0.45,color,times,x,y);

      blue_edge_row(-0.675,-0.225,color,times,x,y);

      blue_edge_row(-0.675,-0.675,color,times,x,y);
      blue_edge_row(-0.45,-0.675,color,times,x,y);
      blue_edge_row(-0.225,-0.675,color,times,x,y);
      blue_edge_row(0.0,-0.675,color,times,x,y);
      blue_edge_row(0.225,-0.675,color,times,x,y);
      blue_edge_row(0.45,-0.675,color,times,x,y);

      blue_edge_col(-0.675,-0.675,color,times,x,y);
      blue_edge_col(-0.675,-0.45,color,times,x,y);

      blue_edge_col(-0.45,-0.225,color,times,x,y);
      blue_edge_col(-0.45,0.0,color,times,x,y);
      blue_edge_col(-0.45,0.225,color,times,x,y);

      blue_edge_col(0.225,0.45,color,times,x,y);

      blue_edge_col(0.675,-0.675,color,times,x,y);
      blue_edge_col(0.675,-0.45,color,times,x,y);
      blue_edge_col(0.675,-0.225,color,times,x,y);
      blue_edge_col(0.675,0.0,color,times,x,y);
      blue_edge_col(0.675,0.225,color,times,x,y);
      blue_edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:
    {
      blue_edge_row(-0.45,0.675,color,times,x,y);
      blue_edge_row(-0.675,0.675,color,times,x,y);

      blue_edge_row(0.225,0.45,color,times,x,y);
      blue_edge_row(0.0,0.45,color,times,x,y);
      blue_edge_row(-0.225,0.45,color,times,x,y);

      blue_edge_row(0.45,-0.225,color,times,x,y);

      blue_edge_row(0.45,-0.675,color,times,x,y);
      blue_edge_row(0.225,-0.675,color,times,x,y);
      blue_edge_row(0.0,-0.675,color,times,x,y);
      blue_edge_row(-0.225,-0.675,color,times,x,y);
      blue_edge_row(-0.45,-0.675,color,times,x,y);
      blue_edge_row(-0.675,-0.675,color,times,x,y);

      blue_edge_col(0.675,-0.675,color,times,x,y);
      blue_edge_col(0.675,-0.45,color,times,x,y);

      blue_edge_col(0.45,-0.225,color,times,x,y);
      blue_edge_col(0.45,0.0,color,times,x,y);
      blue_edge_col(0.45,0.225,color,times,x,y);

      blue_edge_col(-0.225,0.45,color,times,x,y);

      blue_edge_col(-0.675,-0.675,color,times,x,y);
      blue_edge_col(-0.675,-0.45,color,times,x,y);
      blue_edge_col(-0.675,-0.225,color,times,x,y);
      blue_edge_col(-0.675,0.0,color,times,x,y);
      blue_edge_col(-0.675,0.225,color,times,x,y);
      blue_edge_col(-0.675,0.45,color,times,x,y);
    }break;
  case 2:
    {
      blue_edge_row(0.225,-0.675,color,times,x,y);
      blue_edge_row(0.45,-0.675,color,times,x,y);

      blue_edge_row(-0.45,-0.45,color,times,x,y);
      blue_edge_row(-0.225,-0.45,color,times,x,y);
      blue_edge_row(0.0,-0.45,color,times,x,y);

      blue_edge_row(-0.675,0.225,color,times,x,y);

      blue_edge_row(-0.675,0.675,color,times,x,y);
      blue_edge_row(-0.45,0.675,color,times,x,y);
      blue_edge_row(-0.225,0.675,color,times,x,y);
      blue_edge_row(0.0,0.675,color,times,x,y);
      blue_edge_row(0.225,0.675,color,times,x,y);
      blue_edge_row(0.45,0.675,color,times,x,y);

      blue_edge_col(-0.675,0.45,color,times,x,y);
      blue_edge_col(-0.675,0.225,color,times,x,y);

      blue_edge_col(-0.45,0.0,color,times,x,y);
      blue_edge_col(-0.45,-0.225,color,times,x,y);
      blue_edge_col(-0.45,-0.45,color,times,x,y);

      blue_edge_col(0.225,-0.675,color,times,x,y);

      blue_edge_col(0.675,0.45,color,times,x,y);
      blue_edge_col(0.675,0.225,color,times,x,y);
      blue_edge_col(0.675,0.0,color,times,x,y);
      blue_edge_col(0.675,-0.225,color,times,x,y);
      blue_edge_col(0.675,-0.45,color,times,x,y);
      blue_edge_col(0.675,-0.675,color,times,x,y);
    }break;
  case 3:
    {
      blue_edge_row(-0.45,-0.675,color,times,x,y);
      blue_edge_row(-0.675,-0.675,color,times,x,y);

      blue_edge_row(0.225,-0.45,color,times,x,y);
      blue_edge_row(0.0,-0.45,color,times,x,y);
      blue_edge_row(-0.225,-0.45,color,times,x,y);

      blue_edge_row(0.45,0.225,color,times,x,y);

      blue_edge_row(0.45,0.675,color,times,x,y);
      blue_edge_row(0.225,0.675,color,times,x,y);
      blue_edge_row(0.0,0.675,color,times,x,y);
      blue_edge_row(-0.225,0.675,color,times,x,y);
      blue_edge_row(-0.45,0.675,color,times,x,y);
      blue_edge_row(-0.675,0.675,color,times,x,y);

      blue_edge_col(0.675,0.45,color,times,x,y);
      blue_edge_col(0.675,0.225,color,times,x,y);

      blue_edge_col(0.45,0.0,color,times,x,y);
      blue_edge_col(0.45,-0.225,color,times,x,y);
      blue_edge_col(0.45,-0.45,color,times,x,y);

      blue_edge_col(-0.225,-0.675,color,times,x,y);

      blue_edge_col(-0.675,0.45,color,times,x,y);
      blue_edge_col(-0.675,0.225,color,times,x,y);
      blue_edge_col(-0.675,0.0,color,times,x,y);
      blue_edge_col(-0.675,-0.225,color,times,x,y);
      blue_edge_col(-0.675,-0.45,color,times,x,y);
      blue_edge_col(-0.675,-0.675,color,times,x,y);
    }break;
  }
}

void blue_L(int type, float x, float y, float color, float times) //L Shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      blue_edge_row(-0.675,0.675,color,times,x,y);
      blue_edge_row(-0.45,0.675,color,times,x,y);
      blue_edge_row(-0.225,0.675,color,times,x,y);
      blue_edge_row(0.0,0.675,color,times,x,y);
      blue_edge_row(0.225,0.675,color,times,x,y);
      blue_edge_row(0.45,0.675,color,times,x,y);

      blue_edge_row(0.0,0.0,color,times,x,y);
      blue_edge_row(0.225,0.0,color,times,x,y);
      blue_edge_row(0.45,0.0,color,times,x,y);

      blue_edge_row(-0.675,-0.675,color,times,x,y);
      blue_edge_row(-0.45,-0.675,color,times,x,y);
      blue_edge_row(-0.225,-0.675,color,times,x,y);

      blue_edge_col(-0.675,-0.675,color,times,x,y);
      blue_edge_col(-0.675,-0.45,color,times,x,y);
      blue_edge_col(-0.675,-0.225,color,times,x,y);
      blue_edge_col(-0.675,0.0,color,times,x,y);
      blue_edge_col(-0.675,0.225,color,times,x,y);
      blue_edge_col(-0.675,0.45,color,times,x,y);

      blue_edge_col(0.0,-0.675,color,times,x,y);
      blue_edge_col(0.0,-0.45,color,times,x,y);
      blue_edge_col(0.0,-0.225,color,times,x,y);

      blue_edge_col(0.675,0.0,color,times,x,y);
      blue_edge_col(0.675,0.225,color,times,x,y);
      blue_edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    blue_edge_row(0.45,0.675,color,times,x,y);
    blue_edge_row(0.225,0.675,color,times,x,y);
    blue_edge_row(0.0,0.675,color,times,x,y);
    blue_edge_row(-0.225,0.675,color,times,x,y);
    blue_edge_row(-0.45,0.675,color,times,x,y);
    blue_edge_row(-0.675,0.675,color,times,x,y);

    blue_edge_row(-0.225,0.0,color,times,x,y);
    blue_edge_row(-0.45,0.0,color,times,x,y);
    blue_edge_row(-0.675,0.0,color,times,x,y);

    blue_edge_row(0.45,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);

    blue_edge_col(0.675,-0.675,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,0.0,color,times,x,y);
    blue_edge_col(0.675,0.225,color,times,x,y);
    blue_edge_col(0.675,0.45,color,times,x,y);

    blue_edge_col(0.0,-0.675,color,times,x,y);
    blue_edge_col(0.0,-0.45,color,times,x,y);
    blue_edge_col(0.0,-0.225,color,times,x,y);

    blue_edge_col(-0.675,0.0,color,times,x,y);
    blue_edge_col(-0.675,0.225,color,times,x,y);
    blue_edge_col(-0.675,0.45,color,times,x,y);
  }break;
  case 2:{
    blue_edge_row(-0.675,-0.675,color,times,x,y);
    blue_edge_row(-0.45,-0.675,color,times,x,y);
    blue_edge_row(-0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.45,-0.675,color,times,x,y);

    blue_edge_row(0.0,0.0,color,times,x,y);
    blue_edge_row(0.225,0.0,color,times,x,y);
    blue_edge_row(0.45,0.0,color,times,x,y);

    blue_edge_row(-0.675,0.675,color,times,x,y);
    blue_edge_row(-0.45,0.675,color,times,x,y);
    blue_edge_row(-0.225,0.675,color,times,x,y);

    blue_edge_col(-0.675,0.45,color,times,x,y);
    blue_edge_col(-0.675,0.225,color,times,x,y);
    blue_edge_col(-0.675,0.0,color,times,x,y);
    blue_edge_col(-0.675,-0.225,color,times,x,y);
    blue_edge_col(-0.675,-0.45,color,times,x,y);
    blue_edge_col(-0.675,-0.675,color,times,x,y);

    blue_edge_col(0.0,0.45,color,times,x,y);
    blue_edge_col(0.0,0.225,color,times,x,y);
    blue_edge_col(0.0,0.0,color,times,x,y);

    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.675,color,times,x,y);
  }break;
  case 3:{
    blue_edge_row(0.45,-0.675,color,times,x,y);
    blue_edge_row(0.225,-0.675,color,times,x,y);
    blue_edge_row(0.0,-0.675,color,times,x,y);
    blue_edge_row(-0.225,-0.675,color,times,x,y);
    blue_edge_row(-0.45,-0.675,color,times,x,y);
    blue_edge_row(-0.675,-0.675,color,times,x,y);

    blue_edge_row(-0.225,0.0,color,times,x,y);
    blue_edge_row(-0.45,0.0,color,times,x,y);
    blue_edge_row(-0.675,0.0,color,times,x,y);

    blue_edge_row(0.45,0.675,color,times,x,y);
    blue_edge_row(0.225,0.675,color,times,x,y);
    blue_edge_row(0.0,0.675,color,times,x,y);

    blue_edge_col(0.675,0.45,color,times,x,y);
    blue_edge_col(0.675,0.225,color,times,x,y);
    blue_edge_col(0.675,0.0,color,times,x,y);
    blue_edge_col(0.675,-0.225,color,times,x,y);
    blue_edge_col(0.675,-0.45,color,times,x,y);
    blue_edge_col(0.675,-0.675,color,times,x,y);

    blue_edge_col(0.0,0.45,color,times,x,y);
    blue_edge_col(0.0,0.225,color,times,x,y);
    blue_edge_col(0.0,0.0,color,times,x,y);

    blue_edge_col(-0.675,-0.225,color,times,x,y);
    blue_edge_col(-0.675,-0.45,color,times,x,y);
    blue_edge_col(-0.675,-0.675,color,times,x,y);
  }break;
  }
}

void red_Sq(int type, float x, float y, float color, float times) //Square Shape
{
  switch(type){
  case 0: //New shape is Right
    {
      red_edge_col(0.675,-0.675,color,times,x,y);
      red_edge_col(0.675,-0.45,color,times,x,y);
      red_edge_col(0.675,-0.225,color,times,x,y);
      red_edge_col(0.675,0.0,color,times,x,y);
      red_edge_col(0.675,0.225,color,times,x,y);
      red_edge_col(0.675,0.45,color,times,x,y);

      red_edge_col(-0.675,-0.675,color,times,x,y);
      red_edge_col(-0.675,-0.45,color,times,x,y);
      red_edge_col(-0.675,-0.225,color,times,x,y);
      red_edge_col(-0.675,0.0,color,times,x,y);
      red_edge_col(-0.675,0.225,color,times,x,y);
      red_edge_col(-0.675,0.45,color,times,x,y);

      red_edge_row(0.45,0.675,color,times,x,y);
      red_edge_row(0.225,0.675,color,times,x,y);
      red_edge_row(0.0,0.675,color,times,x,y);
      red_edge_row(-0.225,0.675,color,times,x,y);
      red_edge_row(-0.45,0.675,color,times,x,y);
      red_edge_row(-0.675,0.675,color,times,x,y);

      red_edge_row(0.45,-0.675,color,times,x,y);
      red_edge_row(0.225,-0.675,color,times,x,y);
      red_edge_row(0.0,-0.675,color,times,x,y);
      red_edge_row(-0.225,-0.675,color,times,x,y);
      red_edge_row(-0.45,-0.675,color,times,x,y);
      red_edge_row(-0.675,-0.675,color,times,x,y);
    }break;
  case 1:{ //Nwe shape is Left
    red_edge_col(-0.675,-0.675,color,times,x,y);
    red_edge_col(-0.675,-0.45,color,times,x,y);
    red_edge_col(-0.675,-0.225,color,times,x,y);
    red_edge_col(-0.675,0.0,color,times,x,y);
    red_edge_col(-0.675,0.225,color,times,x,y);
    red_edge_col(-0.675,0.45,color,times,x,y);

    red_edge_col(0.675,-0.675,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,0.0,color,times,x,y);
    red_edge_col(0.675,0.225,color,times,x,y);
    red_edge_col(0.675,0.45,color,times,x,y);

    red_edge_row(-0.675,0.675,color,times,x,y);
    red_edge_row(-0.45,0.675,color,times,x,y);
    red_edge_row(-0.225,0.675,color,times,x,y);
    red_edge_row(0.0,0.675,color,times,x,y);
    red_edge_row(0.225,0.675,color,times,x,y);
    red_edge_row(0.45,0.675,color,times,x,y);

    red_edge_row(-0.675,-0.675,color,times,x,y);
    red_edge_row(-0.45,-0.675,color,times,x,y);
    red_edge_row(-0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.45,-0.675,color,times,x,y);
  }break;
  case 2:{
    red_edge_col(0.675,0.45,color,times,x,y);
    red_edge_col(0.675,0.225,color,times,x,y);
    red_edge_col(0.675,0.0,color,times,x,y);
    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    red_edge_col(-0.675,0.45,color,times,x,y);
    red_edge_col(-0.675,0.225,color,times,x,y);
    red_edge_col(-0.675,0.0,color,times,x,y);
    red_edge_col(-0.675,-0.225,color,times,x,y);
    red_edge_col(-0.675,-0.45,color,times,x,y);
    red_edge_col(-0.675,-0.675,color,times,x,y);

    red_edge_row(0.45,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);
    red_edge_row(-0.225,-0.675,color,times,x,y);
    red_edge_row(-0.45,-0.675,color,times,x,y);
    red_edge_row(-0.675,-0.675,color,times,x,y);

    red_edge_row(0.45,0.675,color,times,x,y);
    red_edge_row(0.225,0.675,color,times,x,y);
    red_edge_row(0.0,0.675,color,times,x,y);
    red_edge_row(-0.225,0.675,color,times,x,y);
    red_edge_row(-0.45,0.675,color,times,x,y);
    red_edge_row(-0.675,0.675,color,times,x,y);
  }break;
  case 3:{
    red_edge_col(-0.675,0.45,color,times,x,y);
    red_edge_col(-0.675,0.225,color,times,x,y);
    red_edge_col(-0.675,0.0,color,times,x,y);
    red_edge_col(-0.675,-0.225,color,times,x,y);
    red_edge_col(-0.675,-0.45,color,times,x,y);
    red_edge_col(-0.675,-0.675,color,times,x,y);
    //edge_row(-0.675,0.675,color,times,x,y);

    red_edge_col(0.675,0.45,color,times,x,y);
    red_edge_col(0.675,0.225,color,times,x,y);
    red_edge_col(0.675,0.0,color,times,x,y);
    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.675,color,times,x,y);

    red_edge_row(-0.675,-0.675,color,times,x,y);
    red_edge_row(-0.45,-0.675,color,times,x,y);
    red_edge_row(-0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.45,-0.675,color,times,x,y);

    red_edge_row(-0.675,0.675,color,times,x,y);
    red_edge_row(-0.45,0.675,color,times,x,y);
    red_edge_row(-0.225,0.675,color,times,x,y);
    red_edge_row(0.0,0.675,color,times,x,y);
    red_edge_row(0.225,0.675,color,times,x,y);
    red_edge_row(0.45,0.675,color,times,x,y);
  }break;
  }
}

void red_New(int type, float x, float y, float color, float times) //New Shape
{
  switch (type){
  case 0:
    {
      red_edge_row(0.225,0.675,color,times,x,y);
      red_edge_row(0.45,0.675,color,times,x,y);

      red_edge_row(-0.45,0.45,color,times,x,y);
      red_edge_row(-0.225,0.45,color,times,x,y);
      red_edge_row(0.0,0.45,color,times,x,y);

      red_edge_row(-0.675,-0.225,color,times,x,y);

      red_edge_row(-0.675,-0.675,color,times,x,y);
      red_edge_row(-0.45,-0.675,color,times,x,y);
      red_edge_row(-0.225,-0.675,color,times,x,y);
      red_edge_row(0.0,-0.675,color,times,x,y);
      red_edge_row(0.225,-0.675,color,times,x,y);
      red_edge_row(0.45,-0.675,color,times,x,y);

      red_edge_col(-0.675,-0.675,color,times,x,y);
      red_edge_col(-0.675,-0.45,color,times,x,y);

      red_edge_col(-0.45,-0.225,color,times,x,y);
      red_edge_col(-0.45,0.0,color,times,x,y);
      red_edge_col(-0.45,0.225,color,times,x,y);

      red_edge_col(0.225,0.45,color,times,x,y);

      red_edge_col(0.675,-0.675,color,times,x,y);
      red_edge_col(0.675,-0.45,color,times,x,y);
      red_edge_col(0.675,-0.225,color,times,x,y);
      red_edge_col(0.675,0.0,color,times,x,y);
      red_edge_col(0.675,0.225,color,times,x,y);
      red_edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:
    {
      red_edge_row(-0.45,0.675,color,times,x,y);
      red_edge_row(-0.675,0.675,color,times,x,y);

      red_edge_row(0.225,0.45,color,times,x,y);
      red_edge_row(0.0,0.45,color,times,x,y);
      red_edge_row(-0.225,0.45,color,times,x,y);

      red_edge_row(0.45,-0.225,color,times,x,y);

      red_edge_row(0.45,-0.675,color,times,x,y);
      red_edge_row(0.225,-0.675,color,times,x,y);
      red_edge_row(0.0,-0.675,color,times,x,y);
      red_edge_row(-0.225,-0.675,color,times,x,y);
      red_edge_row(-0.45,-0.675,color,times,x,y);
      red_edge_row(-0.675,-0.675,color,times,x,y);

      red_edge_col(0.675,-0.675,color,times,x,y);
      red_edge_col(0.675,-0.45,color,times,x,y);

      red_edge_col(0.45,-0.225,color,times,x,y);
      red_edge_col(0.45,0.0,color,times,x,y);
      red_edge_col(0.45,0.225,color,times,x,y);

      red_edge_col(-0.225,0.45,color,times,x,y);

      red_edge_col(-0.675,-0.675,color,times,x,y);
      red_edge_col(-0.675,-0.45,color,times,x,y);
      red_edge_col(-0.675,-0.225,color,times,x,y);
      red_edge_col(-0.675,0.0,color,times,x,y);
      red_edge_col(-0.675,0.225,color,times,x,y);
      red_edge_col(-0.675,0.45,color,times,x,y);
    }break;
  case 2:
    {
      red_edge_row(0.225,-0.675,color,times,x,y);
      red_edge_row(0.45,-0.675,color,times,x,y);

      red_edge_row(-0.45,-0.45,color,times,x,y);
      red_edge_row(-0.225,-0.45,color,times,x,y);
      red_edge_row(0.0,-0.45,color,times,x,y);

      red_edge_row(-0.675,0.225,color,times,x,y);

      red_edge_row(-0.675,0.675,color,times,x,y);
      red_edge_row(-0.45,0.675,color,times,x,y);
      red_edge_row(-0.225,0.675,color,times,x,y);
      red_edge_row(0.0,0.675,color,times,x,y);
      red_edge_row(0.225,0.675,color,times,x,y);
      red_edge_row(0.45,0.675,color,times,x,y);

      red_edge_col(-0.675,0.45,color,times,x,y);
      red_edge_col(-0.675,0.225,color,times,x,y);

      red_edge_col(-0.45,0.0,color,times,x,y);
      red_edge_col(-0.45,-0.225,color,times,x,y);
      red_edge_col(-0.45,-0.45,color,times,x,y);

      red_edge_col(0.225,-0.675,color,times,x,y);

      red_edge_col(0.675,0.45,color,times,x,y);
      red_edge_col(0.675,0.225,color,times,x,y);
      red_edge_col(0.675,0.0,color,times,x,y);
      red_edge_col(0.675,-0.225,color,times,x,y);
      red_edge_col(0.675,-0.45,color,times,x,y);
      red_edge_col(0.675,-0.675,color,times,x,y);
    }break;
  case 3:
    {
      red_edge_row(-0.45,-0.675,color,times,x,y);
      red_edge_row(-0.675,-0.675,color,times,x,y);

      red_edge_row(0.225,-0.45,color,times,x,y);
      red_edge_row(0.0,-0.45,color,times,x,y);
      red_edge_row(-0.225,-0.45,color,times,x,y);

      red_edge_row(0.45,0.225,color,times,x,y);

      red_edge_row(0.45,0.675,color,times,x,y);
      red_edge_row(0.225,0.675,color,times,x,y);
      red_edge_row(0.0,0.675,color,times,x,y);
      red_edge_row(-0.225,0.675,color,times,x,y);
      red_edge_row(-0.45,0.675,color,times,x,y);
      red_edge_row(-0.675,0.675,color,times,x,y);

      red_edge_col(0.675,0.45,color,times,x,y);
      red_edge_col(0.675,0.225,color,times,x,y);

      red_edge_col(0.45,0.0,color,times,x,y);
      red_edge_col(0.45,-0.225,color,times,x,y);
      red_edge_col(0.45,-0.45,color,times,x,y);

      red_edge_col(-0.225,-0.675,color,times,x,y);

      red_edge_col(-0.675,0.45,color,times,x,y);
      red_edge_col(-0.675,0.225,color,times,x,y);
      red_edge_col(-0.675,0.0,color,times,x,y);
      red_edge_col(-0.675,-0.225,color,times,x,y);
      red_edge_col(-0.675,-0.45,color,times,x,y);
      red_edge_col(-0.675,-0.675,color,times,x,y);
    }break;
  }
}

void red_L(int type, float x, float y, float color, float times) //L Shape
{
  switch (type){ //L shape is Left
  case 0:
    {
      red_edge_row(-0.675,0.675,color,times,x,y);
      red_edge_row(-0.45,0.675,color,times,x,y);
      red_edge_row(-0.225,0.675,color,times,x,y);
      red_edge_row(0.0,0.675,color,times,x,y);
      red_edge_row(0.225,0.675,color,times,x,y);
      red_edge_row(0.45,0.675,color,times,x,y);

      red_edge_row(0.0,0.0,color,times,x,y);
      red_edge_row(0.225,0.0,color,times,x,y);
      red_edge_row(0.45,0.0,color,times,x,y);

      red_edge_row(-0.675,-0.675,color,times,x,y);
      red_edge_row(-0.45,-0.675,color,times,x,y);
      red_edge_row(-0.225,-0.675,color,times,x,y);

      red_edge_col(-0.675,-0.675,color,times,x,y);
      red_edge_col(-0.675,-0.45,color,times,x,y);
      red_edge_col(-0.675,-0.225,color,times,x,y);
      red_edge_col(-0.675,0.0,color,times,x,y);
      red_edge_col(-0.675,0.225,color,times,x,y);
      red_edge_col(-0.675,0.45,color,times,x,y);

      red_edge_col(0.0,-0.675,color,times,x,y);
      red_edge_col(0.0,-0.45,color,times,x,y);
      red_edge_col(0.0,-0.225,color,times,x,y);

      red_edge_col(0.675,0.0,color,times,x,y);
      red_edge_col(0.675,0.225,color,times,x,y);
      red_edge_col(0.675,0.45,color,times,x,y);
    }break;
  case 1:{ //L shape is Right
    red_edge_row(0.45,0.675,color,times,x,y);
    red_edge_row(0.225,0.675,color,times,x,y);
    red_edge_row(0.0,0.675,color,times,x,y);
    red_edge_row(-0.225,0.675,color,times,x,y);
    red_edge_row(-0.45,0.675,color,times,x,y);
    red_edge_row(-0.675,0.675,color,times,x,y);

    red_edge_row(-0.225,0.0,color,times,x,y);
    red_edge_row(-0.45,0.0,color,times,x,y);
    red_edge_row(-0.675,0.0,color,times,x,y);

    red_edge_row(0.45,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);

    red_edge_col(0.675,-0.675,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,0.0,color,times,x,y);
    red_edge_col(0.675,0.225,color,times,x,y);
    red_edge_col(0.675,0.45,color,times,x,y);

    red_edge_col(0.0,-0.675,color,times,x,y);
    red_edge_col(0.0,-0.45,color,times,x,y);
    red_edge_col(0.0,-0.225,color,times,x,y);

    red_edge_col(-0.675,0.0,color,times,x,y);
    red_edge_col(-0.675,0.225,color,times,x,y);
    red_edge_col(-0.675,0.45,color,times,x,y);
  }break;
  case 2:{
    red_edge_row(-0.675,-0.675,color,times,x,y);
    red_edge_row(-0.45,-0.675,color,times,x,y);
    red_edge_row(-0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.45,-0.675,color,times,x,y);

    red_edge_row(0.0,0.0,color,times,x,y);
    red_edge_row(0.225,0.0,color,times,x,y);
    red_edge_row(0.45,0.0,color,times,x,y);

    red_edge_row(-0.675,0.675,color,times,x,y);
    red_edge_row(-0.45,0.675,color,times,x,y);
    red_edge_row(-0.225,0.675,color,times,x,y);

    red_edge_col(-0.675,0.45,color,times,x,y);
    red_edge_col(-0.675,0.225,color,times,x,y);
    red_edge_col(-0.675,0.0,color,times,x,y);
    red_edge_col(-0.675,-0.225,color,times,x,y);
    red_edge_col(-0.675,-0.45,color,times,x,y);
    red_edge_col(-0.675,-0.675,color,times,x,y);

    red_edge_col(0.0,0.45,color,times,x,y);
    red_edge_col(0.0,0.225,color,times,x,y);
    red_edge_col(0.0,0.0,color,times,x,y);

    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.675,color,times,x,y);
  }break;
  case 3:{
    red_edge_row(0.45,-0.675,color,times,x,y);
    red_edge_row(0.225,-0.675,color,times,x,y);
    red_edge_row(0.0,-0.675,color,times,x,y);
    red_edge_row(-0.225,-0.675,color,times,x,y);
    red_edge_row(-0.45,-0.675,color,times,x,y);
    red_edge_row(-0.675,-0.675,color,times,x,y);

    red_edge_row(-0.225,0.0,color,times,x,y);
    red_edge_row(-0.45,0.0,color,times,x,y);
    red_edge_row(-0.675,0.0,color,times,x,y);

    red_edge_row(0.45,0.675,color,times,x,y);
    red_edge_row(0.225,0.675,color,times,x,y);
    red_edge_row(0.0,0.675,color,times,x,y);

    red_edge_col(0.675,0.45,color,times,x,y);
    red_edge_col(0.675,0.225,color,times,x,y);
    red_edge_col(0.675,0.0,color,times,x,y);
    red_edge_col(0.675,-0.225,color,times,x,y);
    red_edge_col(0.675,-0.45,color,times,x,y);
    red_edge_col(0.675,-0.675,color,times,x,y);

    red_edge_col(0.0,0.45,color,times,x,y);
    red_edge_col(0.0,0.225,color,times,x,y);
    red_edge_col(0.0,0.0,color,times,x,y);

    red_edge_col(-0.675,-0.225,color,times,x,y);
    red_edge_col(-0.675,-0.45,color,times,x,y);
    red_edge_col(-0.675,-0.675,color,times,x,y);
  }break;
  }
}

void Display_FixationBlue(void)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 1.0);

  glVertex2d(-0.025, -0.025);
  glVertex2d( 0.025, -0.025);
  glVertex2d( 0.025,  0.025);
  glVertex2d(-0.025,  0.025);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 1.0);
  glVertex2d(-0.025, -0.025);
  glVertex2d( 0.025, -0.025);
  glVertex2d( 0.025,  0.025);
  glVertex2d(-0.025,  0.025);
  glEnd();

  glPopMatrix();
}

void Display_FixationRed(void)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d( 1.0, 0.0, 0.0);

  glVertex2d(-0.025, -0.025);
  glVertex2d( 0.025, -0.025);
  glVertex2d( 0.025,  0.025);
  glVertex2d(-0.025,  0.025);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d( 1.0, 0.0, 0.0);
  glVertex2d(-0.025, -0.025);
  glVertex2d( 0.025, -0.025);
  glVertex2d( 0.025,  0.025);
  glVertex2d(-0.025,  0.025);
  glEnd();

  glPopMatrix();
}


void Display_Fixation(void)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 0.0);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();

  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 0.0);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();

  glPopMatrix();
}

void Display_CalibPoints(void)
{
  glPushMatrix();

  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 0.0);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();

  glColor3d( 1.0, 1.0, 1.0);
  /*glBegin(GL_POLYGON);
  glVertex2d(-0.775, 0.725);
  glVertex2d(-0.725, 0.725);
  glVertex2d(-0.725, 0.775);
  glVertex2d(-0.775, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, 0.725);
  glVertex2d( 0.775, 0.725);
  glVertex2d( 0.775, 0.775);
  glVertex2d( 0.725, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, -0.775);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.725, -0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -0.775, -0.775);
  glVertex2d( -0.725, -0.775);
  glVertex2d( -0.725, -0.725);
  glVertex2d( -0.775, -0.725);
  glEnd();*/
  glBegin(GL_POLYGON);
  /*glVertex2d(-0.8925, 0.5475);
  glVertex2d(-0.8675, 0.5475);
  glVertex2d(-0.8675, 0.5725);
  glVertex2d(-0.8925, 0.5725);*/
  glVertex2d(-1.260, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.670);
  glVertex2d(-1.260, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.260, 0.645);
  glVertex2d( 1.260, 0.670);
  glVertex2d( 1.235, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, -0.610);
  glVertex2d( 1.260, -0.610);
  glVertex2d( 1.260, -0.585);
  glVertex2d( 1.235, -0.585);
  glEnd();

  glBegin(GL_POLYGON);
  /*glVertex2d( -0.8925, -0.6525);
  glVertex2d( -0.8675, -0.6525);
  glVertex2d( -0.8675, -0.6275);
  glVertex2d( -0.8925, -0.6275);*/
  glVertex2d( -1.26, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.26, -0.585);
  glEnd();

  /*****************************/
  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 0.0);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();
  glColor3d( 1.0, 1.0, 1.0);

  /*glBegin(GL_POLYGON);
  glVertex2d(-0.775, 0.725);
  glVertex2d(-0.725, 0.725);
  glVertex2d(-0.725, 0.775);
  glVertex2d(-0.775, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, 0.725);
  glVertex2d( 0.775, 0.725);
  glVertex2d( 0.775, 0.775);
  glVertex2d( 0.725, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, -0.775);
  glVertex2d( 0.775, -0.775);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.725, -0.725);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -0.775, -0.775);
  glVertex2d( -0.725, -0.775);
  glVertex2d( -0.725, -0.725);
  glVertex2d( -0.775, -0.725);
  glEnd();*/

  glBegin(GL_POLYGON);
  /*glVertex2d(-0.8925, 0.5475);
  glVertex2d(-0.8675, 0.5475);
  glVertex2d(-0.8675, 0.5725);
  glVertex2d(-0.8925, 0.5725);*/
  glVertex2d(-1.26, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.67);
  glVertex2d(-1.26, 0.67);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.26, 0.645);
  glVertex2d( 1.26, 0.67);
  glVertex2d( 1.235, 0.67);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, -0.610);
  glVertex2d( 1.26, -0.610);
  glVertex2d( 1.26, -0.585);
  glVertex2d( 1.235, -0.585);
  glEnd();

  glBegin(GL_POLYGON);
  /*glVertex2d( -0.8925, -0.6525);
  glVertex2d( -0.8675, -0.6525);
  glVertex2d( -0.8675, -0.6275);
  glVertex2d( -0.8925, -0.6275);*/
  glVertex2d( -1.26, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.26, -0.585);
  glEnd();

  glPopMatrix();
}

void Display_PointsF(void)
{
  glPushMatrix();

  glColor3d( 0.0, 0.0, 0.0);

  /*glBegin(GL_POLYGON);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();

  glColor3d( 0.0, 0.0, 0.0);
  glBegin(GL_POLYGON);
  glVertex2d(-0.775, 0.725);
  glVertex2d(-0.725, 0.725);
  glVertex2d(-0.725, 0.775);
  glVertex2d(-0.775, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, 0.725);
  glVertex2d( 0.775, 0.725);
  glVertex2d( 0.775, 0.775);
  glVertex2d( 0.725, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, -0.775);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.725, -0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -0.775, -0.775);
  glVertex2d( -0.725, -0.775);
  glVertex2d( -0.725, -0.725);
  glVertex2d( -0.775, -0.725);
  glEnd();*/
  glBegin(GL_POLYGON);
  glVertex2d(-1.260, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.670);
  glVertex2d(-1.260, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.260, 0.645);
  glVertex2d( 1.260, 0.670);
  glVertex2d( 1.235, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, -0.610);
  glVertex2d( 1.260, -0.610);
  glVertex2d( 1.260, -0.585);
  glVertex2d( 1.235, -0.585);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -1.260, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.260, -0.585);
  glEnd();

  /*****************************/
  glBegin(GL_POLYGON);
  glColor3d( 0.0, 0.0, 0.0);
  glVertex2d(-0.0075, -0.025);
  glVertex2d( 0.0075, -0.025);
  glVertex2d( 0.0075,  0.025);
  glVertex2d(-0.0075,  0.025);
  glEnd();
  glBegin(GL_POLYGON);
  glVertex2d(-0.025, -0.0075);
  glVertex2d(-0.025,  0.0075);
  glVertex2d( 0.025,  0.0075);
  glVertex2d( 0.025, -0.0075);
  glEnd();
  glColor3d( 0.0, 0.0, 0.0);

  /*glBegin(GL_POLYGON);
  glVertex2d(-0.775, 0.725);
  glVertex2d(-0.725, 0.725);
  glVertex2d(-0.725, 0.775);
  glVertex2d(-0.775, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, 0.725);
  glVertex2d( 0.775, 0.725);
  glVertex2d( 0.775, 0.775);
  glVertex2d( 0.725, 0.775);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 0.725, -0.775);
  glVertex2d( 0.775, -0.775);
  glVertex2d( 0.775, -0.725);
  glVertex2d( 0.725, -0.725);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -0.775, -0.775);
  glVertex2d( -0.725, -0.775);
  glVertex2d( -0.725, -0.725);
  glVertex2d( -0.775, -0.725);
  glEnd();*/
  glBegin(GL_POLYGON);
  glVertex2d(-1.260, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.670);
  glVertex2d(-1.260, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.260, 0.645);
  glVertex2d( 1.260, 0.670);
  glVertex2d( 1.235, 0.670);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, -0.610);
  glVertex2d( 1.260, -0.610);
  glVertex2d( 1.260, -0.585);
  glVertex2d( 1.235, -0.585);
  glEnd();

  glBegin(GL_POLYGON);
  glVertex2d( -1.260, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.260, -0.585);
  glEnd();

  glPopMatrix();
}

void Display_Calib1(void)
{
  glPushMatrix();
  glBegin(GL_POLYGON);

  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d(-0.0125, 0.645);
  glVertex2d( 0.0125, 0.645);
  glVertex2d( 0.0125, 0.670);
  glVertex2d(-0.0125, 0.670);
  glEnd();

  /*****************************/

  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d(-0.0125, 0.645);
  glVertex2d( 0.0125, 0.645);
  glVertex2d( 0.0125, 0.670);
  glVertex2d(-0.0125, 0.670);
  glEnd();
  glPopMatrix();
}

void Display_Calib2(void)
{
  glPushMatrix();
  glBegin(GL_POLYGON);

  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d(-0.0125,-0.585);
  glVertex2d( 0.0125,-0.585);
  glVertex2d( 0.0125,-0.610);
  glVertex2d(-0.0125,-0.610);
  glEnd();

  /*****************************/

  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  glVertex2d(-0.0125,-0.585);
  glVertex2d( 0.0125,-0.585);
  glVertex2d( 0.0125,-0.610);
  glVertex2d(-0.0125,-0.610);
  glEnd();

  glPopMatrix();
}

void Display_Calib3(void)
{
  glPushMatrix();
  glBegin(GL_POLYGON);

  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d(-1.260,-0.0125);
  glVertex2d(-1.235,-0.0125);
  glVertex2d(-1.235, 0.0125);
  glVertex2d(-1.260, 0.0125);
  glEnd();

  /*****************************/

  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);
  glVertex2d(-1.260,-0.0125);
  glVertex2d(-1.235,-0.0125);
  glVertex2d(-1.235, 0.0125);
  glVertex2d(-1.260, 0.0125);
  glBegin(GL_POLYGON);

  glEnd();

  glPopMatrix();
}

void Display_Calib4(void)
{
  glPushMatrix();
  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d( 1.235,-0.0125);
  glVertex2d( 1.260,-0.0125);
  glVertex2d( 1.260, 0.0125);
  glVertex2d( 1.235, 0.0125);
  glEnd();

  /*****************************/
  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);
  glVertex2d( 1.235,-0.0125);
  glVertex2d( 1.260,-0.0125);
  glVertex2d( 1.260, 0.0125);
  glVertex2d( 1.235, 0.0125);
  glBegin(GL_POLYGON);

  glEnd();

  glPopMatrix();
}

void Display_Calib5(void)
{
  glPushMatrix();
  glBegin(GL_POLYGON);

  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d(-1.260, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.670);
  glVertex2d(-1.260, 0.670);
  glEnd();

  /*****************************/

  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  glVertex2d(-1.260, 0.645);
  glVertex2d(-1.235, 0.645);
  glVertex2d(-1.235, 0.670);
  glVertex2d(-1.260, 0.670);
  glEnd();

  glPopMatrix();
}

void Display_Calib6(void)
{
  glPushMatrix();
  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.260, 0.645);
  glVertex2d( 1.260, 0.670);
  glVertex2d( 1.235, 0.670);
  glEnd();

  /*****************************/
  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  glVertex2d( 1.235, 0.645);
  glVertex2d( 1.260, 0.645);
  glVertex2d( 1.260, 0.670);
  glVertex2d( 1.235, 0.670);
  glEnd();

  glPopMatrix();
}

void Display_Calib7(void)
{
  glPushMatrix();
  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  /*glVertex2d( 1.035, -0.505);
  glVertex2d( 1.060, -0.505);
  glVertex2d( 1.060, -0.480);
  glVertex2d( 1.035, -0.480);*/
  glVertex2d( 1.235, -0.585);
  glVertex2d( 1.260, -0.585);
  glVertex2d( 1.260, -0.610);
  glVertex2d( 1.235, -0.610);
  glEnd();

  /*****************************/
  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  /*glVertex2d( 1.035, -0.505);
  glVertex2d( 1.060, -0.505);
  glVertex2d( 1.060, -0.480);
  glVertex2d( 1.035, -0.480);*/
  glVertex2d( 1.235, -0.585);
  glVertex2d( 1.260, -0.585);
  glVertex2d( 1.260, -0.610);
  glVertex2d( 1.235, -0.610);
  glEnd();

  glPopMatrix();
}

void Display_Calib8(void)
{
  glPushMatrix();
  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d( -1.260, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.260, -0.585);
  glEnd();

  /*****************************/
  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  glVertex2d( -1.260, -0.610);
  glVertex2d( -1.235, -0.610);
  glVertex2d( -1.235, -0.585);
  glVertex2d( -1.260, -0.585);
  glEnd();

  glPopMatrix();
}

void Display_Calib9(void)
{
  glPushMatrix();
  glColor3d( 1.0, 1.0, 1.0);

  glBegin(GL_POLYGON);
  glVertex2d( -0.0125, -0.0125);
  glVertex2d(  0.0125, -0.0125);
  glVertex2d(  0.0125,  0.0125);
  glVertex2d( -0.0125,  0.0125);
  glEnd();

  /*****************************/
  glColor3d( 1.0, 1.0, 1.0);
  glBegin(GL_POLYGON);

  glBegin(GL_POLYGON);
  glVertex2d( -0.0125, -0.0125);
  glVertex2d(  0.0125, -0.0125);
  glVertex2d(  0.0125,  0.0125);
  glVertex2d( -0.0125,  0.0125);
  glEnd();

  glPopMatrix();
}
