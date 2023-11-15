#include "SeaBee/Attentionator.H"

Attentionator::Attentionator(ModelManager& mgr,
        const std::string& descrName,
        const std::string& tagName) :
        ModelComponent(mgr, descrName, tagName),
        itsImage(0, 0, NO_INIT)
        {

        itsEVC = nub::soft_ref<EnvVisualCortex>(new EnvVisualCortex(mgr));
        addSubComponent(itsEVC);
}

Attentionator::~Attentionator() {
}

void Attentionator::updateImage(Image< PixRGB<byte> > image) {
        itsImage = image;
}

Point2D<int> Attentionator::getSalientPoint() {
        Point2D<int> mostSalientPoint;
        Image<float> salFloatMap;
        float salientValue;
        int mapLevel;

        LINFO("Dims: %s", convertToString(itsImage.getDims()).c_str());

        itsEVC->input(itsImage);
        salFloatMap = itsEVC->getVCXmap();
        mapLevel = itsEVC->getMapLevel();
        findMax(salFloatMap, mostSalientPoint, salientValue);
        mostSalientPoint.i = mostSalientPoint.i << mapLevel;
        mostSalientPoint.j = mostSalientPoint.j << mapLevel;

        return mostSalientPoint;
}

Image<PixRGB<byte> > Attentionator::getSaliencyMap() {
        Image<PixRGB<byte> > salMap = itsEVC->getVCXmap();
        if(salMap.initialized())
                return rescaleBilinear(salMap, itsImage.getDims());
        else {
                LINFO("DIDNT WORK\n");
                return Image<PixRGB<byte> >();
                }
}

