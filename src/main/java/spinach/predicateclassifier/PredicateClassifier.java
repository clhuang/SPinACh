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

/**
 * Given a sentence, a PredicateClassifier classifies the predicates of that sentence
 *
 * @author Calvin Huang
 */
public class PredicateClassifier {

    protected final PerceptronClassifier classifier;
    protected final PredicateFeatureGenerator featureGenerator;

    public final static String PREDICATE_LABEL = "predicate";
    public final static String NOT_PREDICATE_LABEL = "not_predicate";

    /**
     * Makes a predicate classifier from a perceptron and a feature generator
     *
     * @param classifier       perceptron used to classify predicates
     * @param featureGenerator feature generator to generate features
     */
    public PredicateClassifier(PerceptronClassifier classifier, PredicateFeatureGenerator featureGenerator) {
        this.classifier = classifier;
        this.featureGenerator = featureGenerator;
    }

    /**
     * Adds a list of predicates to a sentence
     *
     * @param sentence sentence to analyze
     * @return the same sentence with a list of predicates
     */
    public TokenSentenceAndPredicates sentenceWithPredicates(TokenSentence sentence) {
        return sentenceWithPredicates(sentence, false);
    }

    /**
     * Adds a list of predicates to a sentence based on training weights
     *
     * @param sentence sentence to analyze
     * @return the same sentence with a list of predicates
     */
    public TokenSentenceAndPredicates trainingSentenceWithPredicates(TokenSentence sentence) {
        return sentenceWithPredicates(sentence, true);
    }

    private TokenSentenceAndPredicates sentenceWithPredicates(TokenSentence sentence, boolean training) {
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

    /**
     * Generates a dataset from a frameset to be used in training
     *
     * @param frameSet sentence to be analyzed
     * @return dataset with features of sentence
     */
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

    /**
     * Generates a dataset from a set of frames to be used in training
     *
     * @param frameSets sentences to be analyzed
     * @return dataset with features of sentences
     */
    public Dataset<String, String> datasetFrom(List<SemanticFrameSet> frameSets) {
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (SemanticFrameSet frameSet : frameSets)
            dataset.addAll(datasetFrom(frameSet));

        dataset.applyFeatureCountThreshold(3);

        return dataset;
    }

    /**
     * Updates the perceptron using gold and predicted sentences
     *
     * @param predictedSentence predicted sentence
     * @param goldSentence      true sentence
     */
    public void update(TokenSentenceAndPredicates predictedSentence, TokenSentenceAndPredicates goldSentence) {
        Dataset<String, String> dataset = new Dataset<String, String>();

        for (Token t : goldSentence) {

            String goldLabel;
            String predictedLabel;

            if (goldSentence.isPredicate(t))
                goldLabel = PerceptronClassifier.GOLD_LABEL_PREFIX + PredicateClassifier.PREDICATE_LABEL;
            else
                goldLabel = PerceptronClassifier.GOLD_LABEL_PREFIX + PredicateClassifier.NOT_PREDICATE_LABEL;

            if (predictedSentence.isPredicate(t))
                predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + PredicateClassifier.PREDICATE_LABEL;
            else
                predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + PredicateClassifier.NOT_PREDICATE_LABEL;

            BasicDatum<String, String> datum = (BasicDatum<String, String>)
                    featureGenerator.datumFrom(predictedSentence, t);

            datum.addLabel(goldLabel);
            datum.addLabel(predictedLabel);

            dataset.add(datum);
        }

        classifier.manualTrain(dataset);

    }

    /**
     * Resets the perceptron for this classifier so that it can be retrained
     */
    public void reset() {
        classifier.reset();
    }

}
