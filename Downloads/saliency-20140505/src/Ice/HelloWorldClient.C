#include <Ice/Ice.h>
#include "HelloWorld.ice.H"

using namespace std;
using namespace Demo;

int main(int argc, char* argv[]) {
  int status = 0;
  Ice::CommunicatorPtr ic;
  try {
    ic = Ice::initialize(argc, argv);
    Ice::ObjectPrx base = ic->stringToProxy(
        "SimpleHelloWorld:default -p 10000");
    HelloWorldPrx helloWorld = HelloWorldPrx::checkedCast(base);
    if(!helloWorld)
      throw "Invalid proxy";
    if(argc > 1) {
            cout << "Sending String..." << endl;
      helloWorld->printString(argv[1]);
      cout << "String Send" << endl;
    }
    else {
      helloWorld->printString("Default message");
    }
  }
  catch (const Ice::Exception& ex) {
    cerr << "ERROR CAUGHT!" << ex << endl;
    status = 1;
  }
  catch(const char* msg) {
    cerr << "ERROR CAUGHT!" << msg << endl;
    status = 1;
  }
  if (ic)
    ic->destroy();
  return status;
}
