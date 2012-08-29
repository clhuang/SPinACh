package spinach.argumentclassifier.featuregen;

import spinach.classify.Metric;
import spinach.classify.StructuredClassifier;
import spinach.sentence.SemanticFrameSet;

import java.util.List;

public class FeatureGenTrainer {

    private StructuredClassifier classifier;
    private ExtensibleOnlineFeatureGenerator featureGenerator;
    private List<SemanticFrameSet> goldFrames;

    private double previousF1 = 0;

    public FeatureGenTrainer(StructuredClassifier classifier,
                             List<SemanticFrameSet> goldFrames) {
        this.classifier = classifier;

        ArgumentFeatureGenerator argFeatureGen = classifier.getArgumentClassifier().getFeatureGenerator();

        if (argFeatureGen instanceof ExtensibleOnlineFeatureGenerator)
            featureGenerator = (ExtensibleOnlineFeatureGenerator) argFeatureGen;
        else
            throw new IllegalArgumentException("Cannot train an untrainable Argument Feature Generator");

        this.goldFrames = goldFrames;
    }

    public void train() {
        featureGenerator.clearFeatures();

        classifier.train(goldFrames);
        previousF1 = (new Metric(classifier, goldFrames)).argumentF1s().getCount(Metric.TOTAL);

        for (int i = 0; i < featureGenerator.numFeatureTypes(); i++) {
            featureGenerator.addFeatureType(i);
            classifier.train(goldFrames);

            Metric metric = new Metric(classifier, goldFrames);
            double F1 = metric.argumentF1s().getCount(Metric.TOTAL);

            if (F1 > previousF1)
                previousF1 = F1;
            else
                featureGenerator.removeFeatureType(i);

        }
    }

}
