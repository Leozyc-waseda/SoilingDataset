#include <Ice/Ice.h>
#include "HelloWorld.ice.H"

using namespace std;
using namespace Demo;

class HelloWorldI: public HelloWorld{
public:
    virtual void printString(const string& s,
                             const Ice::Current&);
};

void
HelloWorldI::
printString(const string& s, const Ice::Current&)
{
    cout << s << endl;
}

int
main(int argc, char* argv[])
{
   int status = 0;
    Ice::CommunicatorPtr ic;
    try {
      ic = Ice::initialize(argc, argv);
      Ice::ObjectAdapterPtr adapter =
        ic->createObjectAdapterWithEndpoints(
            "SimpleHelloWorldAdapter", "default -p 10000");
      Ice::ObjectPtr object = new HelloWorldI;
      adapter->add(object, ic->stringToIdentity("SimpleHelloWorld"));
      adapter->activate();
      ic->waitForShutdown();
    } catch (const Ice::Exception& e) {
        cerr << e << endl;
        status = 1;
    } catch (const char* msg) {
        cerr << msg << endl;
        status = 1;
    }
    if (ic) {
        try {
          ic->destroy();
} catch (const Ice::Exception& e) {
            cerr << e << endl;
            status = 1;
        }
    }
    return status;
}


