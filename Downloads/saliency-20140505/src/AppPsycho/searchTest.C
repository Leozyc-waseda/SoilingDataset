#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL/SDL.h>
#include "BO/functionsTest.H"
//#include "functions.h"
//#include "draw.h"
#include "BO/defineConj.H"
/////////////#include "BO/drawConj.H"

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "rutz/time.h"

int type;
int trial;
int record = 1;
int /*start = 0,*/ tracking = 0;
int target[STIM*4];
float color[16];
float hor[16] = {0.25,0.25,-0.25,-0.25, //x location of stimuli
                 -0.25,0.25,0.75,0.75,
                 0.75,0.75,0.25,-0.25,
                 -0.75,-0.75,-0.75,-0.75};
float ver[16] = {0.25,-0.25,-0.25,0.25, // y location of stimuli
                 0.75,0.75,0.75,0.25,
                 -0.25,-0.75,-0.75,-0.75,
                 -0.75,-0.25,0.25,0.75};

//time_t timer;
double Rec_ES[100],Rec_RT[100];
int place[100][16];
int calib[9] = {0,0,0,0,0,0,0,0,0};
int state_calib=0,flag_state=0;
int input,occlude=0;
int target_BO,target_Shape,target_stim,target_type,target_top,target_location;
char subj[128];
/*int BO_L[2][2] = {{1,3},{0,2}};
  int BO_Sq[2][2] = {{0,2},{1,3}};*/
//time_t sec, min;
time_t start_sec,start_min,last_sec,last_min;
clock_t SS,ES;

 // Instantiate a ModelManager: this manages all are little modules
//ModelManager manager("SearchConjunction");

  // Instantiate our various ModelComponents: the event log and eye tracker stuff
//nub::soft_ref<EyeTrackerConfigurator>
//etc(new EyeTrackerConfigurator(manager));

//nub::soft_ref<EventLog> el(new EventLog(manager));
//manager.addSubComponent(el);

//nub::soft_ref<EyeTracker> et = etc->getET();
//et->setEventLog(el);

//nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));


int Require_Result(void)
{
  int i,j;
  FILE *fp1,*fp2;
  char File_Time[128],File_Loc[128];

  /* Make & Open record file. */
  sprintf(File_Time,"%s_RT.data",subj);
  sprintf(File_Loc,"%s_Loc.data",subj);
  printf("test1\n");
  /* Write Answer */
  fp1=fopen(File_Time,"w");
  fp2=fopen(File_Loc,"w");
  printf("test2\n");
  for(i=0;i<trial;i++){
    fprintf(fp1,"%d\t%d\t%6.3f\t%6.3f\n",i,target[i],Rec_RT[i],Rec_ES[i]);
  }
  printf("test3\n");
  for(i=0;i<trial;i++){
    for(j=0;j<16;j++){
      fprintf(fp2,"%d",place[i][j]);
      if(j==15)fprintf(fp2,"\n");
      else fprintf(fp2,"\t");
    }
  }
  printf("test4\n");
  fclose(fp1);
  fclose(fp2);
  printf("test5\n");
  return 0;
}

void quit_tutorial(int code)
{
  SDL_Quit();
  exit(code);
}

void idle(void)
{
  //glutPostRedisplay();
  if(input == 1)input = 0;
  //printf("idle = %d\n",input);
}

void InputName(char *str)
{
  printf("Subject Name : ");
  scanf("%s",str);
  printf("%s\n",subj);
}

void CheckFormat(int argc)
{
  if(argc!=2 && argc !=3){
    printf("Error\n./StimuliBO FileName\n");
    exit(0);
  }
  if(argc == 3)record = 0;
}

void myReshape(GLsizei w, GLsizei h)
{
  glClearColor(BGC, BGC, BGC, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  /*if(w <= h)
    glOrtho( -1.3, 1.3, -1.0 * (GLfloat)h/(GLfloat)w,   // 1.3 for 18deg if assigned to 15)
         1.0 * (GLfloat)h/(GLfloat)w, -10.0, 10.0);
  else
    glOrtho( -1.3 * (GLfloat)h/(GLfloat)w,
    1.3 * (GLfloat)h/(GLfloat)w, -1.0, 1.0, -10.0, 10.0);*/ //original

  /*
    if(w <= h)
    glOrtho( -1.75, 1.75, -1.0 * (GLfloat)h/(GLfloat)w,   // 1.3 for 18deg if assigned to 15)
    1.0 * (GLfloat)h/(GLfloat)w, -10.0, 10.0);
    else
    glOrtho( -1.75 * (GLfloat)h/(GLfloat)w,
    1.75 * (GLfloat)h/(GLfloat)w, -1.0, 1.0, -10.0, 10.0);
  */

  if(w <= h)
    glOrtho( -3.25, 3.25, -1.0 * (GLfloat)h/(GLfloat)w,   // 1.3 for 18deg if assigned to 15)
             1.0 * (GLfloat)h/(GLfloat)w, -10.0, 10.0);
  else
    glOrtho( -3.25 * (GLfloat)h/(GLfloat)w,
             3.25 * (GLfloat)h/(GLfloat)w, -1.0, 1.0, -10.0, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

}

void Init(int stimuli[], int array[], float color[])
{
  int i;

  for(i = 0 ;i < STIM; i++){
    stimuli[i]=0;
  }

  for(i = 0 ; i < 16; i++) array[i] = 0;

  for(i = 0 ; i < 16; i++){
    color[i] = rand()%2;
  }
}

void setTargetConjunction(int stimuli[],int array[],int target_trial)//target_stim is conbination of Target shapes
{
  struct tm *t_st;
  time_t timer;

  time(&timer);
  t_st = localtime(&timer);

  target_location = (rand()+t_st->tm_sec)%16;
  //pattern = search(target_trial);
  if((target_trial)%2==0)target_BO = 1; //left
  else target_BO = 0; //right

  target_Shape = selection(target_trial);
  fragStimuli(stimuli,target_trial,target_BO);

  array[target_location] = 1;
  place[trial][target_location]=target_trial;
}

/* Display Stimulus */
/*void display(void)
{
  //int target_BO,target_Shape,target_stim,target_type,target_top,target_location;
  int stimuli[STIM],array[16];

  printf("checking et false (display)\n");
  usleep(50000);
  et->track(false);

  if(tracking == 0){
    printf("before start\n");
    //et->track(true);
    et->track(true);
    tracking = 1;
    printf("after start\n");
  }


  if(flag_state == 0)Display_Calibration(state_calib,calib);
  else{
    Init(stimuli,array,color);
    //printf("trial = %d\n",trial);
    printf("target[%d] = %d\n",trial,target[trial]);
    target_stim = target[trial];
    target_type = target_stim%4;
    setTargetConjunction(stimuli,array,target[trial]);

    //array[target_location] = 1;

    Display_Conjunction(target_BO,target_Shape,target_stim,target_type,target_location,stimuli,array,place,trial,hor,ver,color);

    SS = clock();
  }
}*/

/*void display(void)
{
  //int target_BO,target_Shape,target_stim,target_type,target_top,target_location;
  int stimuli[STIM],array[16];

  //printf("checking et false (display)\n");
  //usleep(50000);
  //et->track(false);

  //if(tracking == 0){
    // printf("before start\n");
    //et->track(true);
    //et->track(true);
    //tracking = 1;
    //printf("after start\n");
  //}


  if(flag_state == 0){
    //printf("calib\n");
    Display_Calibration(state_calib,calib);
  }
  else{
    Init(stimuli,array,color);
    //printf("trial = %d\n",trial);
    printf("target[%d] = %d\n",trial,target[trial]);
    target_stim = target[trial];
    target_type = target_stim%4;
    setTargetConjunction(stimuli,array,target[trial]);

    //array[target_location] = 1;

    Display_Conjunction(target_BO,target_Shape,target_stim,target_type,target_location,stimuli,array,place,trial,hor,ver,color);

    SS = clock();
  }
}*/

void display(void)
{
  //int target_BO,target_Shape,target_stim,target_type,target_top,target_location;
  int stimuli[STIM],array[16];

  Init(stimuli,array,color);
  //printf("trial = %d\n",trial);
  printf("target[%d] = %d\n",trial,target[trial]);
  target_stim = target[trial];
  target_type = target_stim%4;
  setTargetConjunction(stimuli,array,target[trial]);

  //array[target_location] = 1;

  Display_Conjunction(target_BO,target_Shape,target_stim,target_type,target_location,stimuli,array,place,trial,hor,ver,color);

  SS = clock();

}

/* For Format */
void init(void)
{
  glClearColor(BGC, BGC, BGC, 0.0);
}

/* Exit Process of Key board*/
void keyboard2(SDL_keysym* keysym)
{
  double RT;
  printf("checking et (keyboard)\n");
  //et->track(true);
  if(flag_state == 0){
    switch (keysym->sym){ //exit when 'q', 'Q' or "ESC"
    case SDLK_ESCAPE:
      {
        //printf("before stop\n");
        //usleep(50000);
        //et->track(false);
        //tracking = 0;
        //printf("after stop\n");
        //printf("before final\n");
        //manager.stop();printf("after final\n");
        quit_tutorial(0);
      }

    case SDLK_SPACE:
      {
        state_calib++;
        if(state_calib==13){
          printf("calibration is over\n");
          glClearColor(BGC, BGC, BGC, 0);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          Display_Blank();
          SDL_GL_SwapBuffers();
          msleep(F_SLEEP*5);
          flag_state = 1;
        }
      }
    default:
      break;
    }
  }else{
    switch (keysym->sym){ //exit when 'q', 'Q' or "ESC"
    case SDLK_ESCAPE:
      {
        printf("before stop\n");
        //usleep(50000);
        //et->track(false);
        tracking = 0;
        //printf("after stop\n");
        //printf("before final\n");
        //manager.stop();printf("after final\n");
        quit_tutorial(0);
      }

    case SDLK_SPACE:
      {
        ES=clock();
        RT=(double)(ES-SS)/CLOCKS_PER_SEC;
        Rec_ES[trial]=(double)ES/CLOCKS_PER_SEC;
        Rec_RT[trial]=RT;
        printf("time = %6.3f sec\n",Rec_ES[trial]);
        printf("Reaction time R = %6.3f sec\n",Rec_RT[trial]);
        trial++;
        //printf("time = %6.3f sec\n",(double)ES/CLOCKS_PER_SEC);
        //printf("Reaction time R = %6.3f sec\n",RT);
        glClearColor(BGC, BGC, BGC, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        SDL_GL_SwapBuffers();
        msleep(1000);
      }
    default:
      break;
    }

    if(trial == STIM*4){
      if(record == 1)Require_Result();
      printf("before stop\n");
      //usleep(50000);
      //et->track(false);
      tracking = 0;
      //printf("after stop\n");
      //printf("before final\n");
      //manager.stop();printf("after final\n");
      quit_tutorial(0);
    }
  }
  //printf("before stop\n");
  //usleep(50000);
  //et->track(false);
  //usleep(50000);
  //et->track(false);
  tracking = 0;
  //printf("after stop\n\n");
  SDL_GL_SwapBuffers();
  //glutPostRedisplay();
}

void keyboard(int key)
{
  double RT;
  printf("checking et (keyboard)\n");


  switch (key){ //exit when 'q', 'Q' or "ESC"
  case 'q':
  case 'Q':
  case '\033':
    {
      quit_tutorial(0);
    }

  case ' ':
    {
      ES=clock();
      RT=(double)(ES-SS)/CLOCKS_PER_SEC;
      Rec_ES[trial]=(double)ES/CLOCKS_PER_SEC;
      Rec_RT[trial]=RT;
      printf("time = %6.3f sec\n",Rec_ES[trial]);
      printf("Reaction time R = %6.3f sec\n",Rec_RT[trial]);
      trial++;
        //printf("time = %6.3f sec\n",(double)ES/CLOCKS_PER_SEC);
        //printf("Reaction time R = %6.3f sec\n",RT);
      glClearColor(BGC, BGC, BGC, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      SDL_GL_SwapBuffers();
      msleep(1000);
    }
  default:
    break;
  }

  /*if(trial == 4){//STIM*4){
    if(record == 1)Require_Result();
    quit_tutorial(0);
    }*/

  //glutPostRedisplay();
}

void process_events(void)
{
  SDL_Event event;

  while(SDL_PollEvent(&event)){
    switch(event.type){
    case SDL_KEYDOWN:
      keyboard2(&event.key.keysym);
      break;
    case SDL_QUIT:
      quit_tutorial(0);
      break;
    }
  }
}

void ourInit(void)
{


  glClearColor(BGC,BGC,BGC,0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  // glOrtho(-1.2,1.2,-1.0,1.0,-1.0,1.0); //


  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_POLYGON_SMOOTH);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);//GL_DONT_CARE);


}

void myInit(int target[])
{
  int i;

  for(i=0;i<STIM*4;i++){
    if(i<STIM)target[i]=i;
    else if(i>=STIM && i<2*STIM)target[i]=i-STIM;
    else if(i>=2*STIM && i<3*STIM)target[i]=i-2*STIM;
    else if(i>=3*STIM && i<4*STIM)target[i]=i-3*STIM;
    //printf("%d\n",target[i]);
  }

  shuffle(target,STIM*4);
}

/*void new_mouse(int button, int state,int x, int y)
{
  double RT;

  if(flag_state == 0){
    switch(button){
      case GLUT_RIGHT_BUTTON:
        if(state == GLUT_DOWN){
          state_calib++;
          if(state_calib==13){
            glClearColor(BGC,BGC,BGC,0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Display_Blank();
            glutSwapBuffers();
            msleep(F_SLEEP*5);
            flag_state = 1;
          }
        }
        break;

    case GLUT_LEFT_BUTTON:
      if(state == GLUT_DOWN){
          state_calib++;
          if(state_calib==13){
            glClearColor(BGC,BGC,BGC,0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            Display_Blank();
            glutSwapBuffers();
            msleep(F_SLEEP*5);
            flag_state = 1;
          }
        }
      break;

    default:
      break;
    }
  }else{
    switch(button){
    case GLUT_RIGHT_BUTTON:
      if(state == GLUT_DOWN)
        {
          ES=clock();
          RT=(double)(ES-SS)/CLOCKS_PER_SEC;
          Rec_ES[trial]=(double)ES/CLOCKS_PER_SEC;
          Rec_RT[trial]=RT;
          printf("time = %6.3f sec\n",Rec_ES[trial]);
          printf("Reaction time R = %6.3f sec\n",Rec_RT[trial]);
          trial++;
          //printf("time = %6.3f sec\n",(double)ES/CLOCKS_PER_SEC);
          //printf("Reaction time R = %6.3f sec\n",RT);
          glClearColor(BGC, BGC, BGC, 0);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glutSwapBuffers();
          msleep(1000);
        }
      break;

    case GLUT_LEFT_BUTTON:
      if(state == GLUT_DOWN)
        {
          ES=clock();
          RT=(double)(ES-SS)/CLOCKS_PER_SEC;
          Rec_ES[trial]=(double)ES/CLOCKS_PER_SEC;
          Rec_RT[trial]=RT;
          printf("time = %6.3f sec\n",Rec_ES[trial]);
          printf("Reaction time R = %6.3f sec\n",Rec_RT[trial]);
          trial++;
          //printf("time = %6.3f sec\n",(double)ES/CLOCKS_PER_SEC);
          //printf("Reaction time L = %6.3f sec\n",RT);
          glClearColor(BGC, BGC, BGC, 0);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glutSwapBuffers();
          msleep(1000);
        }
      break;

    default:
      break;
    }
  }
  if(trial == 4){
    if(record == 1)Require_Result();
    exit(0);
  }
  glutPostRedisplay();
}*/

int main(int argc, char *argv[])
{
  const SDL_VideoInfo* info = NULL;
  int c;
  int width = 0;
  int height = 0;
  int bpp = 0;
  uint32 flag = 0;

  MYLOGVERB = LOG_INFO;  // suppress debug messages

// Instantiate a ModelManager:
  ModelManager manager("Psycho SearchConjunction");

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  if(SDL_Init(SDL_INIT_VIDEO)<0)quit_tutorial(1);

  info = SDL_GetVideoInfo();
  width = 500;//640;
    height = 500;//480;
  bpp = info->vfmt->BitsPerPixel;

  manager.setOptionValString(&OPT_EventLogFileName, "searchConj.log");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fileName>", 1, 1) == false)
    return(1);

  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

    // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);



  CheckFormat(argc);
  //printf("test1\n");
  //glutInit(&argc, argv);

  //glClearColor(BGC,BGC,BGC,0);

  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  manager.start();
  el->pushEvent(std::string("===== Trial : Search Conjucntion Task ====="));
  et->calibrate(d);

  d->clearScreen();
  d->clearScreen();
  d->displayText("Experiment start for space");
  d->waitForKey();
  d->clearScreen();

  /*flag =
    SDL_OPENGL |
    SDL_FULLSCREEN |
    SDL_SWSURFACE |
    SDL_HWSURFACE |
    SDL_DOUBLEBUF |
    SDL_HWACCEL;*/

  flag =
    SDL_OPENGL |
    SDL_FULLSCREEN;
  strcpy(subj,argv[1]);
  printf("%s\n",subj);
  myInit(target);
  trial = 0;

  if(SDL_SetVideoMode(width,height,32,flag) == 0)quit_tutorial(1);

  //glClearColor(BGC,BGC,BGC,0);

  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ourInit();

  while(1){
    // start the eye tracker:
    et->track(true);
    //myReshape(width,height);
    display();
    c = d->waitForKey();
    //waitForKey();
    //process_events();
    keyboard(c);
    // stop the eye tracker:
    usleep(50000);
    et->track(false);
    if(trial == 4/*STIM*/)break;
  }


  if(record == 1)Require_Result();
  printf("finish record\n");
  //glutDisplayFunc(display);
  //glutKeyboardFunc(keyboard);
  //glutMouseFunc(new_mouse);

  //glutMouseFunc(new_mouse);
  //glutKeyboardFunc(keyboard);
  //printf("trial = %d\n",trial);
  //glutDisplayFunc(display);

  //glutMainLoop();

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}
