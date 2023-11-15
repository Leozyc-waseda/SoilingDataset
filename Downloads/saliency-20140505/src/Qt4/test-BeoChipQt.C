/*! @file Qt/test-BeoChipQt.C Qt4 interface for BeoChip control */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt4/test-BeoChipQt.C $
// $Id: test-BeoChipQt.C 12277 2009-12-17 07:39:05Z itti $

#include <Qt/qapplication.h>
#include "Qt4/BeoChipMainForm.qt.H"
#include "QtUtil/Util.H"

// ######################################################################
//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener() :
    itsForm(NULL)
  { }

  virtual ~MyBeoChipListener()
  { }

  virtual void setQtForm(BeoChipMainForm *form)
  { itsForm = form; }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    // Note how here we only store the values and do not immediately
    // update the Qt displays. This is because events may come at a
    // very high rate. Rather, we will update the displays at a lower
    // rate from a thread that reads the current values once in a
    // while.
    LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    switch(t)
      {
      case NONE:
        break;

      case PWM0:
        if (itsForm) itsForm->showPWM(0, valint);
        break;

      case PWM1:
        if (itsForm) itsForm->showPWM(1, valint);
        break;

      case KBD:
        if (itsForm)
          for (int i = 0; i < 5; i ++)
            if (valint & (1 << i)) itsForm->showDin(i, true);
            else itsForm->showDin(i, false);
        break;

      case ADC0:
        if (itsForm) itsForm->showAnalog(0, valint);
        break;

      case ADC1:
        if (itsForm) itsForm->showAnalog(1, valint);
        break;

      case RESET:
        if (itsForm) itsForm->beoChipReset();
        LERROR("BeoChip RESET occurred!");
        break;

      case ECHOREP:
        LINFO("BeoChip Echo reply received.");
        break;

      case INOVERFLOW:
        LERROR("BeoChip input overflow!");
        break;

      case SERIALERROR:
        LERROR("BeoChip serial error!");
        break;

      case OUTOVERFLOW:
        LERROR("BeoChip output overflow!");
        break;

      default:
        LERROR("Unknown event %d received!", int(t));
        break;
      }
  }

protected:
  BeoChipMainForm *itsForm; //!< our Qt form
};

// ######################################################################
//! GUI to play with the BeoChip
int main( int argc, const char ** argv )
{
  // instantiate a model manager:
  ModelManager manager("test-BeoChipQt");

  // instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> bc(new BeoChip(manager));
  manager.addSubComponent(bc);

  // let's register our listener:
  rutz::shared_ptr<MyBeoChipListener> lis(new MyBeoChipListener);
  rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  bc->setListener(lis2);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  bc->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // Let's get going:
  manager.start();

  // get the Qt form up and going:
  QApplication a(argc, argv2qt(argc, argv));
  BeoChipMainForm *w = new BeoChipMainForm;

  // let our BeoChipListner know:
  lis->setQtForm(w);

  // get going (will reset the BeoChip):
  w->init(&manager, bc);

  // pass control on to Qt:
  w->show();
  a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));

  // run the form until the user quits it:
  int retval = a.exec();

  // close down the manager:
  manager.stop();

  return retval;
}
