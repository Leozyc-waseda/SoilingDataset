#ifndef SEABEE_ATTENTIONATOR_DEFINED
#define SEABEE_ATTENTIONATOR_DEFINED

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Neuro/EnvVisualCortex.H"
#include "Image/MathOps.H"
#include "Image/Point2D.H"
#include "Image/ShapeOps.H"
#include "rutz/shared_ptr.h"
#include "Component/ModelComponent.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/OptionManager.H"

class Attentionator : public ModelComponent{
        public:
                Attentionator(ModelManager& mgr,
                        const std::string& descrName = "Attentionator",
                        const std::string& tagName = "Attentionator");
                ~Attentionator();
                void updateImage(Image< PixRGB<byte> > image);
                Point2D<int> getSalientPoint();
                Image<PixRGB<byte> > getSaliencyMap();
        private:
                Image< PixRGB<byte> > itsImage;
                ModelManager itsModelManager;
                nub::soft_ref<EnvVisualCortex> itsEVC;
};

#endif
