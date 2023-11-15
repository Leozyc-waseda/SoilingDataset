#include "Devices/SimpleMotor.H"
#include "Component/ModelManager.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Component/ModelOptionDef.H"


int main(const int argc, const char** argv) {
        ModelManager *mgr = new ModelManager("TestSimpleMotor");

  nub::soft_ref<SimpleMotor> motor(new SimpleMotor(*mgr));
  mgr->addSubComponent(motor);

  //mgr->setOptionValString(&OPT_DevName,  "/dev/ttyUSB0");

  mgr->parseCommandLine(argc, argv, "", 0,0);

  mgr->start();

  int speed = 0;

  while(1)
  {
    std::cout << "Enter a speed: ";
    std::cin >> speed;
    motor->setMotor(speed);
  }
}
