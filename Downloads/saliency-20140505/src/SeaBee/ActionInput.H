// #ifndef __ACTION_INPUT_H__
// #define __ACTION_INPUT_H__


// #include "Util/Angle.H"
// #include "Point3D.H"
// #include "rutz/shared_ptr.h"

// #ifndef __SENSOR_INPUT_H__
// #define __SENSOR_INPUT_H__

// class SensorInput {
// public:
//   SensorInput();
//   SensorInput(double d, Angle ang);
//   ~SensorInput();

//   double data;
//   Angle angle;

// };

// #endif

// SensorInput::SensorInput() : data(-1.0), angle(Angle(-1)) {}
// SensorInput::SensorInput(double d, Angle ang) : data(d), angle(ang) {}
// SensorInput::~SensorInput() {}

// #ifndef __VISION_INPUT_H__
// #define __VISION_INPUT_H__

// class VisionInput {
// public:
//   VisionInput();
//   VisionInput(rutz::shared_ptr<Point3D> p, rutz::shared_ptr<Angle> ang);
//   ~VisionInput();

//   rutz::shared_ptr<Point3D> position;
//   rutz::shared_ptr<Angle> angle;
// };

// #endif

// VisionInput::VisionInput() : position(NULL), angle(NULL) {}
// VisionInput::VisionInput(rutz::shared_ptr<Point3D> p, rutz::shared_ptr<Angle> ang) : position(p), angle(ang) {}
// VisionInput::~VisionInput() {}


// #endif
