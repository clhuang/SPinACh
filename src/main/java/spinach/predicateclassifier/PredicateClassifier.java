package spinach.predicateclassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class PredicateClassifier {

    protected final PerceptronClassifier classifier;
    protected final PredicateFeatureGenerator featureGenerator;

    public final static String PREDICATE_LABEL = "predicate";
    public final static String NOT_PREDICATE_LABEL = "not_predicate";

    public PredicateClassifier(PerceptronClassifier classifier, PredicateFeatureGenerator featureGenerator) {
        this.classifier = classifier;
        this.featureGenerator = featureGenerator;
    }

    public TokenSentenceAndPredicates sentenceWithPredicates(TokenSentence sentence) {
        return sentenceWithPredicates(sentence, false);
    }

    public TokenSentenceAndPredicates trainingSentenceWithPredicates(TokenSentence sentence) {
        return sentenceWithPredicates(sentence, true);
    }

    public TokenSentenceAndPredicates sentenceWithPredicates(TokenSentence sentence, boolean training) {
        TokenSentenceAndPredicates sentenceAndPredicates = new TokenSentenceAndPredicates(sentence);
        for (Token t : sentenceAndPredicates) {
            String predicateClass;
            if (training)
                predicateClass = classifier.trainingClassOf(featureGenerator.datumFrom(sentenceAndPredicates, t));
            else
                predicateClass = classifier.classOf(featureGenerator.datumFrom(sentenceAndPredicates, t));
            if (predicateClass.equals(PREDICATE_LABEL))
                sentenceAndPredicates.addPredicate(t);
        }

        return sentenceAndPredicates;
    }

    public Dataset<String, String> datasetFrom(SemanticFrameSet frameSet) {
        Dataset<String, String> dataset = new Dataset<String, String>();
        Set<Token> predicates = new HashSet<Token>(frameSet.getPredicateList());
        for (Token t : frameSet) {
            BasicDatum<String, String> datum = (BasicDatum<String, String>) featureGenerator.datumFrom(frameSet, t);
            if (predicates.contains(t))
                datum.setLabel(PREDICATE_LABEL);
            else
                datum.setLabel(NOT_PREDICATE_LABEL);
            dataset.add(datum);
        }

        return dataset;
    }

    public Dataset<String, String> datasetFrom(List<SemanticFrameSet> frameSets) {
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (SemanticFrameSet frameSet : frameSets)
            dataset.addAll(datasetFrom(frameSet));

        dataset.applyFeatureCountThreshold(3);

        return dataset;
    }

    public void update(SemanticFrameSet predictedFrame, SemanticFrameSet goldFrame) {
        Dataset<String, String> dataset = new Dataset<String, String>();

        for (Token t : goldFrame) {

            String goldLabel;
            String predictedLabel;

            if (goldFrame.isPredicate(t))
                goldLabel = PerceptronClassifier.GOLD_LABEL_PREFIX + PredicateClassifier.PREDICATE_LABEL;
            else
                goldLabel = PerceptronClassifier.GOLD_LABEL_PREFIX + PredicateClassifier.NOT_PREDICATE_LABEL;

            if (predictedFrame.isPredicate(t))
                predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + PredicateClassifier.PREDICATE_LABEL;
            else
                predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + PredicateClassifier.NOT_PREDICATE_LABEL;

            BasicDatum<String, String> datum = (BasicDatum<String, String>)
                    featureGenerator.datumFrom(predictedFrame, t);

            datum.addLabel(goldLabel);
            datum.addLabel(predictedLabel);

            dataset.add(datum);
        }

        classifier.manualTrain(dataset);

    }

}
