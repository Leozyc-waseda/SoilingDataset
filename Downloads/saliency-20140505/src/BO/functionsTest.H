#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL/SDL.h>

#include "drawTest.H"
#include "defineConj.H"

int target_Pt3[2] = {4,6};
int target_Pt2[4] = {1,2,3,5};
int BO_L[2][2] = {{1,3},{0,2}};
int BO_Sq[2][2] = {{0,2},{1,3}};
int BO_Pt2[2][2] = {{0,2},{1,3}};
int BO_Shapes[2][12] = {{0,1,2,3,7,8,9,10,15,16,19,20},
                        {4,5,6,11,12,13,14,17,18,21,22,23}};
int target_dir;

int msleep(int ms)
{
  struct timeval timeout;

  timeout.tv_sec = ms / 1000;
  timeout.tv_usec = (ms % 1000) * 1000;
  if (select(0, (fd_set *) 0, (fd_set *) 0, (fd_set *) 0, &timeout) < 0) {
    perror("usleep");
    return -1;
  }
  return 0;
}

int search(int tar)
{
  int i,j;

  for(i=0;i<2;i++){
    for(j=0;j<12;j++){
      if(tar == BO_Shapes[i][j]){
        return(i);
      }
    }
  }
  printf("Error\n tar = %d\n", tar);
  exit(0);
}


int selection(int tar)
{
  if(tar >= 0 && tar < 4)return 0; //target is Square
  else if(tar >=4 && tar < 8)return 1; //target is New
  else return 2; //target is L shape
}

void shuffle (int array[],int count)
{
  int i,tmp,change,array2[400];

  srand((unsigned int)time(NULL));

  for (i = 0; i < count; ++i)
    array2[i]=array[i];

  for (i = 0; i < count; ++i) {
    change = rand() % count;
    tmp = array2[i];
    array2[i] = array2[change];
    array2[change] = tmp;

  }

  for( i = 0; i < count; i++ ){
    array[i]=array2[i];
  }
}

void fragStimuli(int stimuli[],int target_stim,int target_BO)
{

  if(target_stim >= 0 && target_stim < 4){ //target is Square1
    //for(i=0;i<4;i++)stimuli[i]=3;
    if (target_BO == 0){
      target_dir = 0;
      stimuli[1]=3;
      stimuli[3]=3;
    }else{
      target_dir = 1;
      stimuli[0]=3;
      stimuli[2]=3;
    }
  }//else if(target_stim >=4 && target_stim < 8){ //target is Square2
  //for(i=4;i<7;i++)stimuli[i][target_type]=3;
  else if(target_stim == 4){target_dir = 1;stimuli[9]=3;}
  else if(target_stim == 5){target_dir = 0;stimuli[8]=3;}
  else if(target_stim == 6){target_dir = 3;stimuli[11]=3;}
  else if(target_stim == 7){target_dir = 2;stimuli[10]=3;}
  else if(target_stim == 8){target_dir = 0;stimuli[5]=3;}
  else if(target_stim == 9){target_dir = 1;stimuli[4]=3;}
  else if(target_stim == 10){target_dir = 2;stimuli[7]=3;}
  else if(target_stim == 11){target_dir = 3;stimuli[6]=3;}
  stimuli[target_stim] = 1;
}

void SelectStimuli(int stimuli, float Lcolor, float x, float y, float times)
{
  float Newcolor;

  if(Lcolor ==1.0) Newcolor = 0.0;
  else Newcolor = 1.0;

  switch (stimuli){
  case 0: //Sq1. based on BO
    {
      draw_dist(1,x,y,Newcolor,times);
    }break; //end of "case 0:"
  case 1: //Sq1. based on BO right
    {
      draw_dist(0,x,y,Newcolor,times);
    }break; //end of "case 1:"
  case 2: //Sq1. based on BO right
    {
      draw_dist(3,x,y,Newcolor,times);
    }break; //end of "case 2:"
  case 3: //Sq1. based on BO right
    {
      draw_dist(2,x,y,Newcolor,times);
    }break; //end of "case 3:"
  case 4:
    { //New. based on BO left
      draw_L1(1,x,y,Newcolor,times);
      draw_New1(1,x,y,Lcolor,times);
    }break; //end of "case 4:"
  case 5:// New. based on BO left
    {
      draw_L1(0,x,y,Newcolor,times);
      draw_New1(0,x,y,Lcolor,times);
    }break; //end of "case 5:"
  case 6:// New. based on BO left
    {
      draw_L1(3,x,y,Newcolor,times);
      draw_New1(3,x,y,Lcolor,times);
    }break; //end of "case 6:"
  case 7: // New. based on BO left
    {
      draw_L1(2,x,y,Newcolor,times);
      draw_New1(2,x,y,Lcolor,times);
    }break; //end of "case 7:"
  case 8: // L. based on left
    {
      draw_Square1(0,x,y,Newcolor,times);
      draw_L1(0,x,y,Lcolor,times);
    }break; //end of "case 8:"
  case 9: // L. based on left
    {
      draw_Square1(1,x,y,Newcolor,times);
      draw_L1(1,x,y,Lcolor,times);
    }break; //end of "case 9:"
  case 10: // L. based on left
    {
      draw_Square1(2,x,y,Newcolor,times);
      draw_L1(2,x,y,Lcolor,times);
    }break; //end of "case 10:"
  case 11: // L. based on left
    {
      draw_Square1(3,x,y,Newcolor,times);
      draw_L1(3,x,y,Lcolor,times);
    }break; //end of "case 11:"
  default:
    {
    }break;
  } //end of "switch (stimuli)"

}

void Display_TargetShape(int target,int type)
{
  glPushMatrix();

  switch (target){
  case 0:
    {//void draw_Target_1(int type, float x, float y, float color, float times) //Square Shape
      shape_Sq(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
    }break;
  case 1:
    {
      shape_New(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
    }break;
  case 2:
    {
      shape_L(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
    }break;
  }

  glPopMatrix();
}

void Display_TargetCon(int BO, int target,int type)
{
  glPushMatrix();

  if(BO == 0){//BO right
    switch (target){
    case 0:
      {//void draw_Target_1(int type, float x, float y, float color, float times) //Square Shape
        blue_Sq(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    case 1:
      {
        blue_New(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    default:
      {
        blue_L(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    }
  }else { //BO left
    switch (target){
    case 0:
      {//void draw_Target_1(int type, float x, float y, float color, float times) //Square Shape
        red_Sq(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    case 1:
      {
        red_New(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    default:
      {
        red_L(type, 0.0, 0.0, 1.0, 2.0/(4.0*4.0));
      }break;
    }
  }

  glPopMatrix();
}

void Display_Blank(void)
{
  glPushMatrix();

  glPopMatrix();
}

void Display_Con_L(int target_BO, int target_Shape,int target_stim,int target_type,int target_location,
                   int stimuli[], int array[], int place[][16], int trial, float hor[],float ver[],float color[])
     /* target is L shape and BO */
{
  time_t timer;
  struct tm *t_st;
  int x;
  float noise_x,noise_y;
  int wait,anti_BO;
  int dist_location,dist_stim,dist_stim1,dist_stim2,dist_stim3;

  dist_stim1 = 0;dist_stim2 = 0;dist_stim3 = 0;

  time(&timer);
  t_st = localtime(&timer);

  if(target_BO == 0)anti_BO = 1;
  else anti_BO = 0;

  /***** Show only Fixation *****/
  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  Display_TargetCon(target_BO,target_Shape,target_dir);//Display_TargetCon(int BO, int target,int type)
  SDL_GL_SwapBuffers();

  /***** Show Flash Fixation ******/
  for(wait = 0 ; wait < 5 ; wait++){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_TargetCon(target_BO,target_Shape,target_dir);
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Blank();
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
  }

  /*Draw Target*/
  noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;

  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SelectStimuli(target_stim,color[target_location],hor[target_location]+noise_x,ver[target_location]+noise_y,2.0/(4.0*4.0));

  switch (target_stim){
  case (8):
    {
      dist_stim1 = 1;
      dist_stim2 = 5;
      dist_stim3 = 10;
    }break;
  case (9):
    {
      dist_stim1 = 0;
      dist_stim2 = 4;
      dist_stim3 = 11;
    }break;

  case (10):
    {
      dist_stim1 = 3;
      dist_stim2 = 7;
      dist_stim3 = 8;
    }break;
  case (11):
    {
      dist_stim1 = 2;
      dist_stim2 = 6;
      dist_stim3 = 9;
    }break;
  }

  for(x=0;x<4;x++){
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      if(array[dist_location]==0){
        noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
        noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;
        SelectStimuli(dist_stim1,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim1;
        break;
      }
    }
  }

  for(x=0;x<2;x++){
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      if(array[dist_location]==0){
        noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
        noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;
        SelectStimuli(dist_stim2,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim2;
        break;
      }
    }
  }

  for(x=0;x<2;x++){ //New
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%4)+4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0 && dist_stim%2 == target_BO){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  for(x=0;x<4;x++){ //L
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%4)+8;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0 /*&& dist_stim%2 == target_BO*/){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim3,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  for(x=0;x<3;x++){//square
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = (t_st->tm_sec+rand())%4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0 && dist_stim%2 != target_BO){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }
  //Display_TargetCon(target_BO,target_Shape,target_dir);
  SDL_GL_SwapBuffers();
}

void Display_Con_Sq(int target_BO, int target_Shape,int target_stim,int target_type,int target_location,
                    int stimuli[], int array[], int place[][16], int trial, float hor[],float ver[],float color[])
     /* target is L shape and BO */
{
  time_t timer;
  struct tm *t_st;
  int x;
  float noise_x,noise_y;
  int wait;
  int dist_location,dist_stim,anti_BO;


  time(&timer);
  t_st = localtime(&timer);

  if(target_BO == 0)anti_BO = 1;
  else anti_BO = 0;

  /***** Show only Fixation *****/
  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  Display_TargetCon(target_BO,target_Shape,target_type);//Display_TargetCon(int BO, int target,int type)
  SDL_GL_SwapBuffers();


  /***** Show Flash Fixation ******/
  for(wait = 0 ; wait < 5 ; wait++){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_TargetCon(target_BO,target_Shape,target_type);
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Blank();
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
  }

  /*Draw Target*/
  noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;

  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SelectStimuli(target_stim,color[target_location],hor[target_location]+noise_x,ver[target_location]+noise_y,2.0/(4.0*4.0));
  //dist_type = target_type;

  for(x=0;x<5;x++){//square
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = (t_st->tm_sec+rand())%4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  for(x=0;x<5;x++){ //New
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%4)+4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  for(x=0;x<5;x++){ //L
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%4)+8;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0 && dist_stim%2 == target_BO){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  //Display_TargetCon(target_BO,target_Shape,target_type);
  SDL_GL_SwapBuffers();
}

void Display_Con_New(int target_BO, int target_Shape,int target_stim,int target_type,int target_location,
                     int stimuli[], int array[], int place[][16],int trial,float hor[],float ver[],float color[])
     /* target is New shape and BO */
{
  time_t timer;
  struct tm *t_st;
  int x;
  float noise_x,noise_y;
  int i,wait;
  int dist_location,dist_stim1,dist_stim2,anti_BO;

  dist_stim1 = 0;dist_stim2 = 0;

  time(&timer);
  t_st = localtime(&timer);

  if(target_BO == 0)anti_BO = 1;
  else anti_BO = 0;

  /***** Show only Fixation *****/
  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  Display_TargetCon(target_BO,target_Shape,target_dir);//Display_TargetCon(int BO, int target,int type)
  SDL_GL_SwapBuffers();

  /***** Show Flash Fixation ******/
  for(wait = 0 ; wait < 5 ; wait++){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_TargetCon(target_BO,target_Shape,target_dir);
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Blank();
    SDL_GL_SwapBuffers();
    msleep(H_SLEEP);
  }

  /*Draw Target*/
  noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SelectStimuli(target_stim,color[target_location],hor[target_location]+noise_x,ver[target_location]+noise_y,2.0/(4.0*4.0));
  //void SelectStimuli(int stimuli, int type, float Lcolor, float x, float y, float times)

  switch (target_stim){
  case (4):
    {
      dist_stim1 = 9;
    }break;
  case (5):
    {
      dist_stim1 = 8;
    }break;
  case (6):
    {
      dist_stim1 = 11;
    }break;
  case (7):
    {
      dist_stim1 = 10;
    }break;
  }

  for(i = 0 ;i < 4; i++){ //same shape
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      if(array[dist_location]==0){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim1,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim1;
        break;
      }
    }
  }

  for(x=0;x<4;x++){//New
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim2 = 4+(t_st->tm_sec+rand())%4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim2]==0 && dist_stim2%2 != target_BO){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim2,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim2;
        break;
      }
    }
  }

  for(x=0;x<3;x++){//L
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim2 = 8+(t_st->tm_sec+rand())%4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim2]==0 && dist_stim2 != dist_stim1){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim2,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim2;
        break;
      }
    }
  }

  for(x=0;x<4;x++){//square
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim2 = (t_st->tm_sec+rand())%4;
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim2]==0 && dist_stim2 != dist_stim1){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuli(dist_stim2,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim2;
        break;
      }
    }
  }
  //Display_TargetCon(target_BO,target_Shape,target_dir);
  SDL_GL_SwapBuffers();
}

void Display_Conjunction(int target_BO, int target_Shape,int target_stim,int target_type,
                         int target_location, int stimuli[], int array[], int place[][16], int trial,
                         float hor[],float ver[],float color[])
{
  if(target_Shape == 0){ //target is Square
    Display_Con_Sq(target_BO, target_Shape,target_stim,target_type,target_location,stimuli,
                   array,place,trial,hor,ver,color);
  }else if(target_Shape == 1){ //target is Pt2
    Display_Con_New(target_BO, target_Shape,target_stim,target_type,target_location,stimuli,
                    array,place,trial,hor,ver,color);
  }else if(target_Shape == 2){ //target is L shape
    Display_Con_L(target_BO, target_Shape,target_stim,target_type,target_location,stimuli,
                  array,place,trial,hor,ver,color);
  }
}

void Display_Calibration(int state_calib, int calib[9])
{
  int i,check;
  time_t timer;
  struct tm *t_st;

  time(&timer);
  t_st = localtime(&timer);
  //printf("state_calib = %d\n",state_calib);

  if(state_calib == 0 || state_calib == 2){
    //printf("state = %d\n",state_calib);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Fixation();
    SDL_GL_SwapBuffers();
  }
  else if(state_calib == 1){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Display_Blank();
    Display_Fixation();
    SDL_GL_SwapBuffers();
    msleep(F_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_CalibPoints();
    SDL_GL_SwapBuffers();
  }else if(state_calib >= 3 && state_calib <= 11){
    for(;;){
      check = ((t_st->tm_sec+rand())%9);
      if(calib[check]==0){
        calib[check]=1;
        break;
      }
    }

    for(i=0;i<5;i++){
      msleep(CALIB);
      glClearColor(BGC, BGC, BGC, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      Display_Blank();
      SDL_GL_SwapBuffers();
      msleep(CALIB);
      glClearColor(BGC, BGC, BGC, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      Display_Fixation();
      SDL_GL_SwapBuffers();
    }
    //printf("state_calib = %d\n",state_calib);
    switch (check){
    case 0:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //Display_Fixation();
        Display_Calib1();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    case 1:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib2();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    case 2:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib3();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    case 3:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib4();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    case 4:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib5();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
    break;
    }
    case 5:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib6();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
    }
      break;
    }
    case 6:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib7();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    case 7:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib8();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
    }
      break;
    }
    case 8:{
      for(i=0;i<5;i++){
        //msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Calib9();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        SDL_GL_SwapBuffers();
        msleep(CALIB);
      }
      break;
    }
    default:break;
    }
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Display_Blank();
    Display_Fixation();
    SDL_GL_SwapBuffers();
  }else if(state_calib == 12){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Display_Blank();
    Display_Fixation();
    SDL_GL_SwapBuffers();
    msleep(F_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_PointsF();
    SDL_GL_SwapBuffers();
  }
}
