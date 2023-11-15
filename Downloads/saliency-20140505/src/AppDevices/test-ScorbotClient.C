#include <Ice/Ice.h>
#include "Ice/Scorbot.ice.H"
#include "Image/Point2D.H"
#include <signal.h>
#include <ncurses.h>

Robots::ScorbotIcePrx scorbot;
WINDOW *mainWin;

// ######################################################################
void cleanupAndExit()
{
  std::cerr <<
    std::endl << "*** EXITING - STOPPING SCORBOT ***" << std::endl;
  scorbot->stopAllMotors();
  sleep(1);
  endwin();
  exit(0);
}

// ######################################################################
void terminate(int s)
{
  cleanupAndExit();
}

// ######################################################################
void printMenu()
{
    mvprintw(0,0,"Scorbot Remote Control");
    clrtoeol();
    mvprintw(2,0,"P Set Position of Target: ");
    clrtoeol();
    mvprintw(3,0,"ESC quits");
    clrtoeol();

    refresh();
}

Point2D<double> getTargetPos()
{
  double x, y;

  mvprintw(4, 0, "");
  clrtoeol();
  mvprintw(5, 0, "");
  clrtoeol();
  nodelay(mainWin, false);
  echo();
  mvscanw(2, 27, "%f %f", &x, &y);
  clrtoeol();
  mvprintw(4, 0, "Pointing To: %f, %f", -343, y);
  clrtoeol();
  nodelay(mainWin, true);
  noecho();

  return Point2D<double>(x,y);
}

// ######################################################################
// Point the camera at a 2D position on the board at a given angle (camera to board angle, in radians), with the camera
// 'camDist' mm away from the target, and the linear slide set xOffset mm to the side of the target.
Robots::ArmPos calcArmPos(Point2D<double> targetPos, double xOffset, double angle, double camDist)
{
  double x = targetPos.i;
  double y = targetPos.j;

  Robots::ArmPos pos;

  // Slide offset
  pos.ex1 = x + xOffset;

  // Base Angle
  pos.base = atan2(y,x);

  //Distance to the target from the base axis
  double x_t = sqrt(x*x+y*y);

  //Camera Positions
  double camX = x_t - cos(angle)*camDist;
  double camY = sin(angle)*camDist;

  mvprintw(5, 0, "CamPos: %f, %f", camX, camY);
  refresh();

  return pos;
}

// ######################################################################
void mainLoop()
{
  mainWin = initscr();
  mainWin = mainWin;
  noecho();
  cbreak();

  nodelay(mainWin, true);
  //keypad(win, true);

  printMenu();
  while(1)
  {
    char c = getch();

    switch(c)
    {
      case 27: //Escape Key
        cleanupAndExit();
        break;
      case 'p':
      case 'P':
        Point2D<double> targetPos = getTargetPos();
        Robots::ArmPos armPos = calcArmPos(targetPos, 0, M_PI/4, 500);

        break;

    }

    printMenu();
    
//    mvprintw(3, 0, "You Pressed: %d  ", c);
    clrtoeol();
    refresh();
    wrefresh(mainWin);

  }

}

// ######################################################################
int main(int argc, char* argv[])
{

	signal(SIGHUP, terminate); signal(SIGINT, terminate);
	signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
	signal(SIGALRM, terminate);
  
  int status = 0;
  Ice::CommunicatorPtr ic;

  try {
    ic = Ice::initialize(argc, argv);
    Ice::ObjectPrx base = ic->stringToProxy(
        "ScorbotServer:default -p 10000");
    scorbot = Robots::ScorbotIcePrx::checkedCast(base);
    if(!scorbot)
      throw "Invalid Proxy!";

    mainLoop();

  } catch(const Ice::Exception& ex) {
    std::cerr << ex << std::endl;
    status = 1;
  } catch(const char* msg) {
    std::cerr << msg << std::endl;
    status = 1;
  }

  if(ic)
    ic->destroy();

  cleanupAndExit();
  return status;
}
