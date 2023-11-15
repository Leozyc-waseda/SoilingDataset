#include "drawBO.H"
#include "defineBO.H"

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

void fragStimuliBO(int stimuli[],int target_stim,int target_BO)
{
  int i;

  if (target_stim==0){
    for (i=0;i<STIM;i++){
      if(i!=1 && i!=3 && i!=9)stimuli[i]=3;
    }
  }else if (target_stim==1){
    for (i=0;i<STIM;i++){
      if(i!=0 && i!=2 && i!=8)stimuli[i]=3;
    }
  }else if (target_stim==2){
    for (i=0;i<STIM;i++){
      if(i!=1 && i!=3 && i!=11)stimuli[i]=3;
    }
  }else if (target_stim==3){
    for (i=0;i<STIM;i++){
      if(i!=0 && i!=2 && i!=10)stimuli[i]=3;
    }
  }else if (target_stim==4){
    for (i=0;i<STIM;i++){
      if(i!=9)stimuli[i]=3;
    }
  }else if (target_stim==5){
    for (i=0;i<STIM;i++){
      if(i!=8)stimuli[i]=3;
    }
  }else if (target_stim==6){
    for (i=0;i<STIM;i++){
      if(i!=11)stimuli[i]=3;
    }
  }else if (target_stim==7){
    for (i=0;i<STIM;i++){
      if(i!=10)stimuli[i]=3;
    }
  }else if (target_stim==8){
    for (i=0;i<STIM;i++){
      if(i!=1 && i!=5)stimuli[i]=3;
    }
  }else if (target_stim==9){
    for (i=0;i<STIM;i++){
      if(i!=0 && i!=4)stimuli[i]=3;
    }
  }else if (target_stim==10){
    for (i=0;i<STIM;i++){
      if(i!=3 && i!=7)stimuli[i]=3;
    }
  }else if (target_stim==11){
    for (i=0;i<STIM;i++){
      if(i!=2 && i!=6)stimuli[i]=3;
    }
  }

  stimuli[target_stim] = 1;
}

void SelectStimuliBO(int stimuli, float Lcolor, float x, float y, float times)
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
  case 12: //Sq1. based on BO
    {
      draw_dist(1,x,y,Newcolor,times);
    }break; //end of "case 12:"
  case 13: //Sq1. based on BO right
    {
      draw_dist(0,x,y,Newcolor,times);
    }break; //end of "case 13:"
  case 14: //Sq1. based on BO right
    {
      draw_dist(3,x,y,Newcolor,times);
    }break; //end of "case 14:"
  case 15: //Sq1. based on BO right
    {
      draw_dist(2,x,y,Newcolor,times);
    }break; //end of "case 15:"
  case 16:
    { //New. based on BO left
      draw_L1(1,x,y,Newcolor,times);
      draw_New1(1,x,y,Lcolor,times);
    }break; //end of "case 16:"
  case 17:// New. based on BO left
    {
      draw_L1(0,x,y,Newcolor,times);
      draw_New1(0,x,y,Lcolor,times);
    }break; //end of "case 17:"
  case 18:// New. based on BO left
    {
      draw_L1(3,x,y,Newcolor,times);
      draw_New1(3,x,y,Lcolor,times);
    }break; //end of "case 18:"
  case 19: // New. based on BO left
    {
      draw_L1(2,x,y,Newcolor,times);
      draw_New1(2,x,y,Lcolor,times);
    }break; //end of "case 19:"
  case 20: // L. based on left
    {
      draw_Square1(0,x,y,Newcolor,times);
      draw_L1(0,x,y,Lcolor,times);
    }break; //end of "case 20:"
  case 21: // L. based on left
    {
      draw_Square1(1,x,y,Newcolor,times);
      draw_L1(1,x,y,Lcolor,times);
    }break; //end of "case 21:"
  case 22: // L. based on left
    {
      draw_Square1(2,x,y,Newcolor,times);
      draw_L1(2,x,y,Lcolor,times);
    }break; //end of "case 22:"
  case 23: // L. based on left
    {
      draw_Square1(3,x,y,Newcolor,times);
      draw_L1(3,x,y,Lcolor,times);
    }break; //end of "case 23:"
  default:
    {
    }break;
  } //end of "switch (stimuli)"
}

void Display_Blank(void)
{
  glPushMatrix();

  glPopMatrix();
}

void Display_BORight(int target_BO,int target_stim,int target_location,int stimuli[],
                     int array[], int place[][16], int trial, float hor[],float ver[],float color[])
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
  Display_FixationBlue();//Display_TargetCon(int BO, int target,int type)
  glutSwapBuffers();

  /***** Show Flash Fixation ******/
  for(wait = 0 ; wait < 5 ; wait++){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_FixationBlue();
    glutSwapBuffers();
    msleep(H_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Blank();
    glutSwapBuffers();
    msleep(H_SLEEP);
  }

  /*Draw Target*/
  noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;

  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SelectStimuliBO(target_stim,color[target_location],hor[target_location]+noise_x,ver[target_location]+noise_y,2.0/(4.0*4.0));


  for(x=0;x<15;x++){ //New
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%STIM);
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuliBO(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }


  //Display_TargetCon(target_BO,target_Shape,target_dir);
  glutSwapBuffers();
}

void Display_BOLeft(int target_BO,int target_stim,int target_location, int stimuli[],
                    int array[], int place[][16],int trial,float hor[],float ver[],float color[])
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
  Display_FixationRed();
  glutSwapBuffers();

   /***** Show Flash Fixation ******/
  for(wait = 0 ; wait < 5 ; wait++){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_FixationRed();
    glutSwapBuffers();
    msleep(H_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Blank();
    glutSwapBuffers();
    msleep(H_SLEEP);
  }

  /*Draw Target*/
  noise_x = ((t_st->tm_sec+rand())%11-5.0)/100.0;
  noise_y = ((t_st->tm_sec+rand())%11-5.0)/100.0;

  glClearColor(BGC, BGC, BGC, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SelectStimuliBO(target_stim,color[target_location],hor[target_location]+noise_x,ver[target_location]+noise_y,2.0/(4.0*4.0));

  for(x=0;x<15;x++){
    for(;;){
      dist_location = (t_st->tm_sec+rand())%16;
      dist_stim = ((t_st->tm_sec+rand())%STIM);
      //dist_type = (t_st->tm_sec+rand())%TYPE;
      if(array[dist_location]==0 && stimuli[dist_stim]==0){
        noise_x = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        noise_y = (((t_st->tm_sec+rand())%11)-5.0)/100.0;
        SelectStimuliBO(dist_stim,color[dist_location],hor[dist_location]+noise_x,ver[dist_location]+noise_y,2.0/(4.0*4.0));
        array[dist_location]=2;
        place[trial][dist_location]=dist_stim;
        break;
      }
    }
  }

  //Display_TargetCon(target_BO,target_Shape,target_dir);
  glutSwapBuffers();
}



void Display_BO(int target_BO,int target_stim,int target_type,int target_location,
                int stimuli[], int array[], int place[][16],int trial,float hor[],float ver[],float color[])
{
  if(target_BO == 0){ //target is BO Right
    Display_BORight(target_BO,target_stim,target_location,stimuli,array,place,trial,hor,ver,color);
  }else if(target_BO == 1){ //target is BO Left
    Display_BOLeft(target_BO,target_stim,target_location,stimuli,array,place,trial,hor,ver,color);
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
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_Fixation();
    glutSwapBuffers();
  }
  else if(state_calib == 1){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Display_Blank();
    Display_Fixation();
    glutSwapBuffers();
    msleep(F_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_CalibPoints();
    glutSwapBuffers();
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
      glutSwapBuffers();
      msleep(CALIB);
      glClearColor(BGC, BGC, BGC, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      Display_Fixation();
      glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
        glutSwapBuffers();
        msleep(CALIB);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Display_Blank();
        //Display_Fixation();
        glutSwapBuffers();
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
    glutSwapBuffers();
  }else if(state_calib == 12){
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Display_Blank();
    Display_Fixation();
    glutSwapBuffers();
    msleep(F_SLEEP);
    glClearColor(BGC, BGC, BGC, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Display_PointsF();
    glutSwapBuffers();
  }
}
