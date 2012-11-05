package test;

import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.LeftRightArgumentClassifier;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.classify.Metric;
import spinach.classify.UnstructuredClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;

import java.io.IOException;
import java.util.List;
import java.util.TreeSet;

import static test.TestConstants.*;

public class UnstructuredTest {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        List<SemanticFrameSet> frameSets = CorpusUtils.parseCorpus(TRAIN_CORPUS);

        ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        argumentFeatureGenerator.reduceFeatureSet(frameSets);
        PerceptronClassifier argumentClassifierPerceptron =
                new PerceptronClassifier(argumentFeatureGenerator.getAllowedNonStructuralFeatures(),
                        ArgumentClassifier.getLabelSet(frameSets), NUM_EPOCHS);

        argumentClassifierPerceptron.setBurnInPeriod(BURN_IN_PERIOD);

        ArgumentClassifier argumentClassifier =
                //StructuredClassifier.importArgumentClassifier("src/test/resources/argumentClassifierB.gz");
                new LeftRightArgumentClassifier(argumentClassifierPerceptron, argumentFeatureGenerator);

        /*PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();
        predicateFeatureGenerator.reduceFeatureSet(frameSets);
        PerceptronClassifier predicateClassifierPerceptron =
                new PerceptronClassifier(predicateFeatureGenerator.getAllowedNonStructuralFeatures(),
                        PredicateClassifier.getLabelSet(), 10);*/
        PredicateClassifier predicateClassifier =
                PredicateClassifier.importClassifier("src/test/resources/predicateClassifierA.gz");
        //new PredicateClassifier(predicateClassifierPerceptron, predicateFeatureGenerator);

        UnstructuredClassifier classifier = new UnstructuredClassifier(argumentClassifier, predicateClassifier, 1, frameSets);

        //classifier.trainPredicateClassifier(frameSets);
        classifier.trainArgumentClassifier();

        argumentClassifier.exportClassifier("src/test/resources/argumentClassifierB.gz");
        //classifier.exportPredicateClassifier("src/test/resources/predicateClassifierA.gz");

        System.out.println("Exported classifiers");

        List<SemanticFrameSet> testFrameSets = CorpusUtils.parseCorpus(DEVEL_CORPUS);
        System.out.println("parsed devel corpus");

        /*classifier.trainArgumentFeatureGenerator(frameSets, testFrameSets);
        for (IndividualFeatureGenerator f :
                ((ExtensibleFeatureGenerator) argumentFeatureGenerator).enabledFeatures())
            System.out.println(f.identifier);*/

        Metric m = new Metric(classifier, testFrameSets);
        m.recalculateScores();

        System.out.format("Predicates %d %d %d\n", m.predicateCorrect(), m.predicatePredicted(), m.predicateGold());

        for (String s : new TreeSet<String>(m.argumentF1s().keySet())) {
            if (!s.equals(Metric.TOTAL))
                System.out.format("%s %.4f %.4f %.4f\n", s, m.argumentPrecisions().getCount(s),
                        m.argumentRecalls().getCount(s), m.argumentF1s().getCount(s));
        }
        String s = Metric.TOTAL;
        System.out.format("%s %.4f %.4f %.4f\n", s, m.argumentPrecisions().getCount(s),
                m.argumentRecalls().getCount(s), m.argumentF1s().getCount(s));

    }
}