// #ifndef __COMPLEX_MOVEMENT_H__
// #define __COMPLEX_MOVEMENT_H__

// #include "Globals.H"
// #include "PreMotorComplex.H"
// #include "ActionInput.H"

// // class PreMotorComplex;


// // typedef void (PreMotorComplex::*FnPtr)(const SensorInput);
// // typedef void (PreMotorComplex::*Vis_FnPtr) (const VisionInput&);

// class ComplexMovement {
// public:

//   ComplexMovement();
//   ~ComplexMovement();

// //   friend class PreMotorComplex;

// //   ////// ADDS A ONCE/REPEAT WORLD SENSOR MOVE //////
// //   void addMove(FnPtr funcPointer,  NumMoves num, const double d);
// //   void addMove(FnPtr funcPointer,  NumMoves num, const Angle a);

// //   ////// ADDS A ONCE/REPEAT VISION SENSOR MOVE //////
// //   void addMove(Vis_FnPtr funcPointer, NumMoves num, const rutz::shared_ptr<Point3D> p);
// //   void addMove(Vis_FnPtr funcPointer, NumMoves num, const rutz::shared_ptr<Angle> a);

// private:



// //   std::vector<FnPtr> W_onceFuncs;
// //   std::vector<SensorInput> W_onceParams;

// //   std::vector<FnPtr> W_reFuncs;
// //   std::vector<SensorInput> W_reParams;


// //   std::vector<Vis_FnPtr> Vis_onceFuncs;
// //   std::vector<VisionInput> Vis_onceParams;

// //   std::vector<Vis_FnPtr> Vis_reFuncs;
// //   std::vector<VisionInput> Vis_reParams;

//   /*enum moveType { CENTER, SWEEP, 360TURN } */

//   // world/sensor moves
// //   void addRepeatMove(FnPtr funcPointer, const double d, const Angle a);
// //   void addOnceMove(FnPtr funcPointer, const double d, const Angle a);

// //   // vision moves
// //   void addRepeatMove(Vis_FnPtr funcPointer, const rutz::shared_ptr<Point3D> p, const rutz::shared_ptr<Angle> a);
// //   void addOnceMove(Vis_FnPtr funcPointer, const rutz::shared_ptr<Point3D> p, const rutz::shared_ptr<Angle> a);



// };




// ComplexMovement::ComplexMovement() {}
// ComplexMovement::~ComplexMovement() {}


// // //////////////// MOVEMENT ADDING FUNCTIONS //////////////
// // void ComplexMovement::addMove(FnPtr funcPointer, NumMoves num, const double d) {

// //  //    if(num == ONCE)
// // //       addOnceMove(funcPointer, d, Angle(-1));
// // //     else if(num == REPEAT)
// // //       addRepeatMove(funcPointer, d, Angle(-1));
// // //     else
// // //       LINFO("DID NOT ADD MOVE");

// // }

// // void ComplexMovement::addMove(FnPtr funcPointer, NumMoves num, const Angle a) {

// // //     if(num == ONCE)
// // //       addOnceMove(funcPointer, -1.0, a);
// // //     else if(num == REPEAT)
// // //       addRepeatMove(funcPointer, -1.0, a);
// // //     else
// // //       LINFO("DID NOT ADD MOVE");

// // }

// // void ComplexMovement::addMove(Movements move_type, Domain domain, NumMoves num, Point3D* p) {
// // //  if(domain == VISION) {
// // //     if(move_type == TURN) {
// // //       if(num == ONCE)
// // //         addOnceMove(&PreMotorComplex::vturn, p, NULL);
// // //       else
// // //         addRecursiveMove(&PreMotorComplex::vturn, p, NULL);
// // //     }
// // //     else if(move_type == DIVE) {
// // //       if(num == ONCE)
// // //         addOnceMove(&PreMotorComplex::vdive, p, NULL);
// // //       else
// // //         addRecursiveMove(&PreMotorComplex::vdive, p, NULL);
// // //     }
// // //     else if(move_type == FORWARD) {
// // //       if(num == ONCE)
// // //         addOnceMove(&PreMotorComplex::vforward, p, NULL);
// // //       else
// // //         addRecursiveMove(&PreMotorComplex::vforward, p, NULL);
// // //     }
// // //     else
// // //       printf("ComplexMovement::addMove::improper data/domain for VISION MOVE\n");
// // //   }
// // //   else
// // //     LINFO("DID NOT ADD MOVE");
// // }

// // void ComplexMovement::addMove(Vis_FnPtr funcPointer, NumMoves num, const rutz::shared_ptr<Angle> a) {

// // //   rutz::shared_ptr<Point3D> p;
// // //   if(num == ONCE)
// // //     addOnceMove(funcPointer, p, a);
// // //   else if(num == REPEAT)
// // //     addRepeatMove(funcPointer, p, a);
// // //   else
// // //     LINFO("DID NOT ADD MOVE");

// // }





// // ///////////////// WORLD/SENSOR ADD MOVES. ACTUALLY ADDS TO LIST ///////////
// // void ComplexMovement::addRepeatMove(FnPtr funcPointer, const double d, const Angle a) {

// // //   W_reFuncs.push_back(NULL);
// // //   W_reFuncs[W_reFuncs.size() - 1] = funcPointer;
// // //   SensorInput params(d, a);
// // //   W_reParams.push_back(params);

// // }

// // void ComplexMovement::addOnceMove(FnPtr funcPointer, const double d, const Angle a) {

// // //   W_onceFuncs.push_back(NULL);
// // //   W_onceFuncs[W_onceFuncs.size() - 1] = funcPointer;
// // //   SensorInput params(d, a);
// // //   W_onceParams.push_back(params);

// // }


// // ////////////////// VISION ADD MOVES. ACTUALLY ADDS TO LIST /////////////
// // void ComplexMovement::addOnceMove(Vis_FnPtr funcPointer, const rutz::shared_ptr<Point3D> p, const rutz::shared_ptr<Angle> a) {
// // //   Vis_onceFuncs.push_back(NULL);
// // //   Vis_onceFuncs[Vis_onceFuncs.size() - 1] = funcPointer;
// // //   VisionInput params(p, a);
// // //   Vis_onceParams.push_back(params);

// // }

// // void ComplexMovement::addRepeatMove(Vis_FnPtr funcPointer, const rutz::shared_ptr<Point3D> p, const rutz::shared_ptr<Angle> a) {
// // //   Vis_reFuncs.push_back(NULL);
// // //   Vis_reFuncs[Vis_reFuncs.size() - 1] = funcPointer;
// // //   VisionInput params(p, a);
// // //   Vis_reParams.push_back(params);
// // }




// #endif
