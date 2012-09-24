package test;

import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.EasyFirstArgumentClassifier;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.classify.Metric;
import spinach.classify.StructuredClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.predicateclassifier.PredicateFeatureGenerator;
import spinach.sentence.SemanticFrameSet;

import java.util.List;
import java.util.TreeSet;

public class Main {
    public static void main(String[] args) {

        List<SemanticFrameSet> frameSets = CorpusUtils.parseCorpus("src/test/resources/train.closed");

        ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();

        argumentFeatureGenerator.reduceFeatureSet(frameSets);
        predicateFeatureGenerator.reduceFeatureSet(frameSets);

        System.out.println("Generated reduced feature sets");

        PerceptronClassifier argumentClassifierPerceptron =
                new PerceptronClassifier(argumentFeatureGenerator.getAllowedNonStructuralFeatures(), 10);
        PerceptronClassifier predicateClassifierPerceptron =
                new PerceptronClassifier(predicateFeatureGenerator.getAllowedNonStructuralFeatures(), 10);

        ArgumentClassifier argumentClassifier =
                //StructuredClassifier.importArgumentClassifier("src/test/resources/argumentClassifierA.gz");
                new EasyFirstArgumentClassifier(argumentClassifierPerceptron, argumentFeatureGenerator);

        System.out.println("imported argument classifier");

        PredicateClassifier predicateClassifier =
                //StructuredClassifier.importPredicateClassifier("src/test/resources/predicateClassifierA.gz");
                new PredicateClassifier(predicateClassifierPerceptron, predicateFeatureGenerator);

        System.out.println("imported classifiers");

        StructuredClassifier classifier = new StructuredClassifier(argumentClassifier, predicateClassifier, 10);

        classifier.train(frameSets);

        classifier.exportArgumentClassifier("src/test/resources/argumentClassifierA.gz");
        classifier.exportPredicateClassifier("src/test/resources/predicateClassifierA.gz");

        System.out.println("Exported classifiers");

        List<SemanticFrameSet> testFrameSets = CorpusUtils.parseCorpus("src/test/resources/devel.closed");
        System.out.println("parsed devel corpus");

        Metric m = new Metric(classifier, testFrameSets);

        System.out.format("Predicates %.4f %.4f %.4f\n", m.predicateRecall(), m.predicatePrecision(), m.predicateF1());

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
