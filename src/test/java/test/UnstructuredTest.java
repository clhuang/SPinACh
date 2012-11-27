package test;

import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.EasyFirstArgumentClassifier;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.classify.Metric;
import spinach.classify.UnstructuredClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.predicateclassifier.PredicateFeatureGenerator;
import spinach.sentence.SemanticFrameSet;

import java.io.IOException;
import java.util.List;

import static test.TestConstants.*;

public class UnstructuredTest {
    private static List<SemanticFrameSet> trainingFrames;
    private static final String ARG_CLASSIFIER_LOC = "src/test/resources/argumentClassifierUNS.gz";
    private static final String PRED_CLASSIFIER_LOC = "src/test/resources/predicateClassifierA.gz";

    private static final boolean LOAD_ARG_CLASSIFIER = false;
    private static final boolean LOAD_PRED_CLASSIFIER = true;

    private static final int BURN_IN_PERIOD = 700;

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        trainingFrames = CorpusUtils.parseCorpus(TRAIN_CORPUS);
        System.out.println("Parsed train corpus");

        ArgumentClassifier argumentClassifier = LOAD_ARG_CLASSIFIER ?
                ArgumentClassifier.importClassifier(ARG_CLASSIFIER_LOC) :
                initArgClassifier();

        argumentClassifier.setConsistencyMode(false, false);

        PredicateClassifier predicateClassifier = LOAD_PRED_CLASSIFIER ?
                PredicateClassifier.importClassifier(PRED_CLASSIFIER_LOC) :
                initPredClassifier();

        UnstructuredClassifier classifier = new UnstructuredClassifier(argumentClassifier, predicateClassifier,
                trainingFrames);

        if (!LOAD_PRED_CLASSIFIER) {
            classifier.trainPredicateClassifier();
            predicateClassifier.exportClassifier(PRED_CLASSIFIER_LOC);
        }
        if (!LOAD_ARG_CLASSIFIER) {
            classifier.trainArgumentClassifier();
            argumentClassifier.exportClassifier(ARG_CLASSIFIER_LOC);
        }

        List<SemanticFrameSet> testFrameSets = CorpusUtils.parseCorpus(DEVEL_CORPUS);
        System.out.println("parsed devel corpus");

        /*classifier.trainArgumentFeatureGenerator(frameSets, testFrameSets);
        for (IndividualFeatureGenerator f :
                ((ExtensibleFeatureGenerator) argumentFeatureGenerator).enabledFeatures())
            System.out.println(f.identifier);*/

        Metric m = new Metric(classifier, testFrameSets);
        m.recalculateScores();

        System.out.format("Predicates %d %d %d\n", m.predicateCorrect(), m.predicatePredicted(), m.predicateGold());

        String s = Metric.TOTAL;
        System.out.format("%s %.4f %.4f %.4f\n", s, m.argumentPrecisions().getCount(s),
                m.argumentRecalls().getCount(s), m.argumentF1s().getCount(s));

    }

    private static ArgumentClassifier initArgClassifier() {
        ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        argumentFeatureGenerator.reduceFeatureSet(trainingFrames);
        System.out.println("Arg classifier reduced feature set");
        PerceptronClassifier argumentClassifierPerceptron =
                new PerceptronClassifier(NUM_EPOCHS);

        argumentClassifierPerceptron.setBurnInPeriod(BURN_IN_PERIOD);

        return new EasyFirstArgumentClassifier(argumentClassifierPerceptron, argumentFeatureGenerator);
    }

    private static PredicateClassifier initPredClassifier() {
        PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();
        predicateFeatureGenerator.reduceFeatureSet(trainingFrames);
        System.out.println("Pred classifier reduced feature set");
        PerceptronClassifier predicateClassifierPerceptron =
                new PerceptronClassifier(predicateFeatureGenerator.getAllowedNonStructuralFeatures(),
                        PredicateClassifier.getLabelSet(), NUM_EPOCHS);
        System.out.println("Created pred classifier perceptron");

        return new PredicateClassifier(predicateClassifierPerceptron, predicateFeatureGenerator);
    }
}
