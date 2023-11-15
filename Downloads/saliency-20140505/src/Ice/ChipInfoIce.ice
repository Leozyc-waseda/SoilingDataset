#include "ImageIce.ice" 
//"



module ChipInfoIceMod {

  sequence<int>    IntSeq;  
  sequence<double> DoubleSeq;

  interface ChipInfoServerIce {
    ["ami"] double update(ImageIceMod::ImageIce image, int chipID);
 
    ImageIceMod::ImageIce getSmap(int chipID);
    DoubleSeq getBelief(int chipID);
    DoubleSeq getBelief2(int chipID);
    DoubleSeq getHist(int chipID);
    

    //Tell a server that it should create, and be responsible for instances of
    //chips whose IDs are in chipIDList
    void registerChips(IntSeq chipIDList);

    //Start running the server. You must do this after registering all of your chips, and before
    //calling the first update, otherwise the memory systems of EnvVisualCortex will not be initialized
    void startServer();


    //Statistics
    //Return the avg number of chips per second
    float getAvgChipsSec(); 
  };

  interface ChipInfoServerFactoryIce {
    ChipInfoServerIce* createServer();
  };
};
