#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <GL/gl.h>
#undef APIENTRY // otherwise it gets redefined between gl.h and glut.h???
#include <GL/glut.h>
#include "BO/functionsConj.H"
//#include "functions.h"
//#include "draw.h"
#include "BO/defineConj.H"
/////////////#include "BO/drawConj.H"

#include "Component/ModelManager.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"

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
ModelManager manager("SearchConjunction");

  // Instantiate our various ModelComponents: the event log and eye tracker stuff
nub::soft_ref<EyeTrackerConfigurator>
etc(new EyeTrackerConfigurator(manager));

nub::soft_ref<EventLog> el(new EventLog(manager));
//manager.addSubComponent(el);

nub::soft_ref<EyeTracker> et;
//et->setEventLog(el);

void Require_Record(void)
{
  int i,j;
  FILE *fp1,*fp2;
  char File_Time[256],File_Loc[256];

  /* Make & Open record file. */
  //sprintf(File_Time,"/lab/nobuhiko/Record/%s_RT.data",subj);
  //sprintf(File_Loc,"/lab/nobuhiko/Record/%s_Loc.data",subj);
   sprintf(File_Time,"%s_RT.data",subj);
   sprintf(File_Loc,"%s_Loc.data",subj);

  /* Write Answer */
  fp1=fopen(File_Time,"w");if(fp1==NULL)printf("error");
  fp2=fopen(File_Loc,"w");

  for(i=0;i<trial;i++){
    fprintf(fp1,"%d\t%d\t%6.3f\t%6.3f\n",i,target[i],Rec_RT[i],Rec_ES[i]);
  }

  for(i=0;i<trial;i++){
    for(j=0;j<16;j++){
      fprintf(fp2,"%d",place[i][j]);
      if(j==15)fprintf(fp2,"\n");
      else fprintf(fp2,"\t");
    }
  }
  fclose(fp1);
  fclose(fp2);
  //return 0;
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
  if (scanf("%s",str) != 1) LFATAL("error in scanf");
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
void display(void)
{
  //int target_BO,target_Shape,target_stim,target_type,target_top,target_location;
  int stimuli[STIM],array[16];


  /*et->track(false);
   if(tracking == 0){
    printf("before start\n");
    //et->track(true);
    et->track(true);
    tracking = 1;
    printf("after start\n");
    }*/

  if(flag_state == 0){
    if(tracking == 0){
      if(state_calib>=3 && state_calib <= 11){
        et->track(true);
        tracking = 1;
      }
    }
    Display_Calibration(state_calib,calib);
    if(tracking == 1){
      usleep(50000);
        et->track(false);
        tracking = 0;
    }
  }else{
    Init(stimuli,array,color);
    printf("trial = %d\n",trial);
    //printf("before start\n");
        //et->track(true);
    et->track(true);
    tracking = 1;
    //printf("after start\n");
    printf("target[%d] = %d\n",trial,target[trial]);
    target_stim = target[trial];
    target_type = target_stim%4;
    setTargetConjunction(stimuli,array,target[trial]);

    //array[target_location] = 1;

    Display_Conjunction(target_BO,target_Shape,target_stim,target_type,target_location,stimuli,array,place,trial,hor,ver,color);

    SS = clock();
  }
}

/* For Format */
void init(void)
{
  glClearColor(BGC, BGC, BGC, 0.0);
}

/* Exit Process of Key board*/
void keyboard(unsigned char key,int x,int y)
{
  double RT;

  //printf("checking et (keyboard)\n");
  //et->track(true);
  if(flag_state == 0){
    switch (key){ //exit when 'q', 'Q' or "ESC"
    case 'q':
    case 'Q':
    case '\033':
      {
        //printf("before stop\n");
        usleep(50000);
        et->track(false);
        tracking = 0;
        //printf("after stop\n");
        //printf("before final\n");
        manager.stop();
        exit(0);
      }

    case ' ':
      {
        usleep(50000);
        et->track(false);
        tracking = 0;
        state_calib++;
        if(state_calib==13){
          //printf("calibration is over\n");
          glClearColor(BGC, BGC, BGC, 0);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          Display_Blank();
          glutSwapBuffers();
          msleep(F_SLEEP*5);
          flag_state = 1;
        }
      }
    default:
      break;
    }
  }else{
    switch (key){ //exit when 'q', 'Q' or "ESC"
    case 'q':
    case 'Q':
    case '\033':
      {
        //printf("before stop\n");
        usleep(50000);
        et->track(false);
        tracking = 0;
        //printf("after stop\n");
        //printf("before final\n");
        manager.stop();//printf("after final\n");
        exit(0);
      }

    case ' ':
      {
        usleep(50000);
        et->track(false);
        //usleep(50000);
        //et->track(false);
        tracking = 0;
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
    default:
      break;
    }

    if(trial == STIM*4){
      //printf("before stop\n");
      usleep(50000);
      et->track(false);
      tracking = 0;
      //printf("after stop\n");
      //printf("before final\n");
      manager.stop();//printf("after final\n");
      Require_Record();
      exit(0);
    }
  }
  glutPostRedisplay();
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

void new_mouse(int button, int state,int x, int y)
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
    if(record == 1)Require_Record();
    exit(0);
  }
  glutPostRedisplay();
}

int main(int argc, char *argv[])
{
  const char *name = "New Stimulus";

  MYLOGVERB = LOG_INFO;  // suppress debug messages

  //  manager.addSubComponent(etc);
  manager.addSubComponent(etc);
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "searchConj.log");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fileName>", 1, 1) == false)
    return(1);

  et = etc->getET();
  et->setEventLog(el);


  manager.start();
  el->pushEvent(std::string("===== Trial : Search Conjucntion Task ====="));


  //CheckFormat(argc);
  //printf("test1\n");
  glutInit(&argc, argv);
  strcpy(subj,argv[1]);
  //printf("%s\n",subj);

  myInit(target);
  trial = 0;
  //printf("test2\n");
 glClearColor(BGC,BGC,BGC,0);
 //printf("test3\n");
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  //printf("test4\n");
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE); //for movie
  //printf("test5\n");
  glutInitWindowPosition(100,100);
  glutInitWindowSize(500,500);

  //InputOcclude(&occlude);


  //glutInitDisplayMode(GLUT_RGBA);
  glutCreateWindow(name);

  glutFullScreen();

  ourInit();
  glutReshapeFunc(myReshape);
  //glutKeyboardFunc(keyboard);
  //printf("test6\n");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(new_mouse);
  glutIdleFunc(idle);
  //glutMouseFunc(new_mouse);
  //glutKeyboardFunc(keyboard);
  //printf("trial = %d\n",trial);
  //glutDisplayFunc(display);

  glutMainLoop();
  return 0;
}
