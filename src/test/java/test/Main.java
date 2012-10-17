package test;

import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.EasyFirstArgumentClassifier;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.classify.Metric;
import spinach.classify.StructuredClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;

import java.util.List;
import java.util.TreeSet;

public class Main {
    public static void main(String[] args) {

        List<SemanticFrameSet> frameSets = CorpusUtils.parseCorpus("src/test/resources/train.closed");

        ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        argumentFeatureGenerator.reduceFeatureSet(frameSets);
        PerceptronClassifier argumentClassifierPerceptron =
                new PerceptronClassifier(argumentFeatureGenerator.getAllowedNonStructuralFeatures(),
                        ArgumentClassifier.getLabelSet(frameSets), 10);

        argumentClassifierPerceptron.setBurnInPeriod(800000);

        ArgumentClassifier argumentClassifier =
                //StructuredClassifier.importArgumentClassifier("src/test/resources/argumentClassifierA.gz");
                new EasyFirstArgumentClassifier(argumentClassifierPerceptron, argumentFeatureGenerator);

        /*PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();
        predicateFeatureGenerator.reduceFeatureSet(frameSets);
        PerceptronClassifier predicateClassifierPerceptron =
                new PerceptronClassifier(predicateFeatureGenerator.getAllowedNonStructuralFeatures(),
                        PredicateClassifier.getLabelSet(), 10);*/
        PredicateClassifier predicateClassifier =
                StructuredClassifier.importPredicateClassifier("src/test/resources/predicateClassifierA.gz");
        //new PredicateClassifier(predicateClassifierPerceptron, predicateFeatureGenerator);

        StructuredClassifier classifier = new StructuredClassifier(argumentClassifier, predicateClassifier, 10);

        //classifier.trainPredicateClassifier(frameSets);
        classifier.trainArgumentClassifier(frameSets);

        classifier.exportArgumentClassifier("src/test/resources/argumentClassifierB.gz");
        //classifier.exportPredicateClassifier("src/test/resources/predicateClassifierA.gz");

        System.out.println("Exported classifiers");

        List<SemanticFrameSet> testFrameSets = CorpusUtils.parseCorpus("src/test/resources/devel.closed");
        System.out.println("parsed devel corpus");

        /*for (int i = 0; i < predicateClassifierPerceptron.featureIndex.size(); i++)
            System.out.println(predicateClassifierPerceptron.featureIndex.get(i) +
                    " " + predicateClassifierPerceptron.zWeights.get(0).weights[i] +
                    " " + predicateClassifierPerceptron.zWeights.get(0).avgWeights[i] +
                    " " + predicateClassifierPerceptron.zWeights.get(0).lastUpdateIteration[i] +
                    " " + predicateClassifierPerceptron.zWeights.get(0).currentIteration);

        System.exit(0);*/

        Metric m = new Metric(classifier, testFrameSets);

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
