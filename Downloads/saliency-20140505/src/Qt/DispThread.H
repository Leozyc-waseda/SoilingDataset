#include <qthread.h>
#include "Image/Image.H"
#include "Qt/ImageCanvas.h"
#include "Qt/ui/RobotHeadForm.h"


class DispThread : public QThread
{
  public:
    void init(nub::soft_ref<FrameIstream> gb,
        ImageCanvas *rightDisp, ImageCanvas *leftDisp);
    void run();

  private:
    nub::soft_ref<FrameIstream> itsGb;
    ImageCanvas* itsRightDisp;
    ImageCanvas* itsLeftDisp;
    bool stop;

};

void DispThread::init(nub::soft_ref<FrameIstream> gb,
        ImageCanvas *rightDisp, ImageCanvas *leftDisp)
    {
      itsGb = gb;
      itsRightDisp = rightDisp;
      itsLeftDisp = leftDisp;
      stop = false;
    }
void DispThread::run()
{
  printf("Start\n");
  while(!stop)
  {
    Image<PixRGB<byte> > img = itsGb->readRGB();
    //itsRightDisp->setImage(img);
 //   itsLeftDisp->setImage(img);
  }
  printf("Stop\n");
}

