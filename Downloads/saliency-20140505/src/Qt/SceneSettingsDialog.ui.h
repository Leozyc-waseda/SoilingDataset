/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you want to add, delete, or rename functions or slots, use
** Qt Designer to update this file, preserving your code.
**
** You should not define a constructor or destructor in this file.
** Instead, write your code in functions called init() and destroy().
** These will automatically be called by the form's constructor and
** destructor.
*****************************************************************************/


void SceneSettingsDialog::init( SceneGenerator *trainScene, SceneGenerator *testScene )
{
  itsTrainScene = trainScene;
  itsTestScene = testScene;

  static bool setSceneTypeList = true;

  //Set up the lists of scene types
  if (setSceneTypeList)
  {
    for (uint i=0; i<itsTrainScene->getNumSceneTypes(); i++){
      itsTrainingSceneType->insertItem(
          QString(itsTrainScene->getSceneTypeNames(
              SceneGenerator::SceneType(i))));

      itsTestingSceneType->insertItem(
          QString(itsTestScene->getSceneTypeNames(
              SceneGenerator::SceneType(i))));
    }
    setSceneTypeList = false;

  }


  //set up the Train settings
  itsTrainingSceneType->setCurrentItem(itsTrainScene->getSceneType());

  itsTrainingTargetObject->setText(
      QString("%L1").arg(itsTrainScene->getTargetObject()));

  PixRGB<byte> targetColor = itsTrainScene->getTargetColor();
  itsTrainingTargetColorR->setText(QString("%L1").arg(targetColor[0]));
  itsTrainingTargetColorG->setText(QString("%L1").arg(targetColor[1]));
  itsTrainingTargetColorB->setText(QString("%L1").arg(targetColor[2]));

  itsTrainingLum->setText(
      QString("%L1").arg(itsTrainScene->getLum()));

  itsTrainingRotation->setText(
      QString("%L1").arg(itsTrainScene->getRotation()));

  PixRGB<byte> color = itsTrainScene->getColor();
  itsTrainingColorR->setText(QString("%L1").arg(color[0]));
  itsTrainingColorG->setText(QString("%L1").arg(color[1]));
  itsTrainingColorB->setText(QString("%L1").arg(color[2]));

  itsTrainingNoise->setText(
      QString("%L1").arg(itsTrainScene->getNoise()));

  PixRGB<byte> backColor = itsTrainScene->getBackgroundColor();
  itsTrainingBackColorR->setText(QString("%L1").arg(backColor[0]));
  itsTrainingBackColorG->setText(QString("%L1").arg(backColor[1]));
  itsTrainingBackColorB->setText(QString("%L1").arg(backColor[2]));

  //set up the Test settings
  itsTestingSceneType->setCurrentItem(itsTestScene->getSceneType());

  itsTestingTargetObject->setText(
      QString("%L1").arg(itsTestScene->getTargetObject()));

  targetColor = itsTestScene->getTargetColor();
  itsTestingTargetColorR->setText(QString("%L1").arg(targetColor[0]));
  itsTestingTargetColorG->setText(QString("%L1").arg(targetColor[1]));
  itsTestingTargetColorB->setText(QString("%L1").arg(targetColor[2]));

  itsTestingLum->setText(
      QString("%L1").arg(itsTestScene->getLum()));

  color = itsTestScene->getColor();
  itsTestingColorR->setText(QString("%L1").arg(color[0]));
  itsTestingColorG->setText(QString("%L1").arg(color[1]));
  itsTestingColorB->setText(QString("%L1").arg(color[2]));

  itsTestingNoise->setText(
      QString("%L1").arg(itsTestScene->getNoise()));

  backColor = itsTestScene->getBackgroundColor();
  itsTestingBackColorR->setText(QString("%L1").arg(backColor[0]));
  itsTestingBackColorG->setText(QString("%L1").arg(backColor[1]));
  itsTestingBackColorB->setText(QString("%L1").arg(backColor[2]));

  itsTestingRotation->setText(
      QString("%L1").arg(itsTestScene->getRotation()));
}




void SceneSettingsDialog::accept()
{

  //set the train scene;

  itsTrainScene->setSceneType(
      (SceneGenerator::SceneType)itsTrainingSceneType->currentItem());

  itsTrainScene->setTargetObject(
      itsTrainingTargetObject->text().toInt());

  itsTrainScene->setTargetColor(PixRGB<byte>(
    itsTrainingTargetColorR->text().toInt(),
    itsTrainingTargetColorG->text().toInt(),
    itsTrainingTargetColorB->text().toInt()
    ));

  itsTrainScene->setLum(itsTrainingLum->text().toInt());

  itsTrainScene->setRotation(itsTrainingRotation->text().toInt());

  itsTrainScene->setColor(PixRGB<byte>(
    itsTrainingColorR->text().toInt(),
    itsTrainingColorG->text().toInt(),
    itsTrainingColorB->text().toInt()
    ));

  itsTrainScene->setNoise(itsTrainingNoise->text().toInt());

  itsTrainScene->setBackgroundColor(PixRGB<byte>(
    itsTrainingBackColorR->text().toInt(),
    itsTrainingBackColorG->text().toInt(),
    itsTrainingBackColorB->text().toInt()
    ));


  //set the Test scene;

  itsTestScene->setSceneType(
      (SceneGenerator::SceneType)itsTestingSceneType->currentItem());

  itsTestScene->setTargetObject(
      itsTestingTargetObject->text().toInt());

  itsTestScene->setTargetColor(PixRGB<byte>(
    itsTestingTargetColorR->text().toInt(),
    itsTestingTargetColorG->text().toInt(),
    itsTestingTargetColorB->text().toInt()
    ));

  itsTestScene->setLum(itsTestingLum->text().toInt());

  itsTestScene->setRotation(itsTestingRotation->text().toInt());

  itsTestScene->setColor(PixRGB<byte>(
    itsTestingColorR->text().toInt(),
    itsTestingColorG->text().toInt(),
    itsTestingColorB->text().toInt()
    ));

  itsTestScene->setNoise(itsTestingNoise->text().toInt());

  itsTestScene->setBackgroundColor(PixRGB<byte>(
    itsTestingBackColorR->text().toInt(),
    itsTestingBackColorG->text().toInt(),
    itsTestingBackColorB->text().toInt()
    ));


  close();

}
