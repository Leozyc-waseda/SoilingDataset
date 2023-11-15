#include "Point3D.H"
#include "Globals.H"

int main( int argc, const char* argv[] )
{
  Point3D pt1 = Point3D(1,1,1);
  Point3D pt2 = Point3D(2,2,2);

  Point3D sum = pt1 + pt2;

  std::cout<<"Sum X: "<<sum.x<<" Sum Y: "<<sum.y<<" Sum Z: "<<sum.z<<std::endl;

  std::cout<<"Distance between pt1 and sum is: "<<pt1.distance(sum)<<std::endl;

  std::cout<<"Type of sum: "<<typeid(pt1).name()<<std::endl;

  if(typeid(pt1).name() == typeid(pt2).name())
    {
      std::cout<<"pt1 and pt2 have equal types"<<std::endl;
    }

  return 0;
}
