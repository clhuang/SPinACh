package spinach.predicateclassifier;

import com.google.common.collect.ImmutableList;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.io.Serializable;
import java.util.List;

/**
 * Given a sentence, a PredicateClassifier classifies the predicates of that sentence
 *
 * @author Calvin Huang
 */
public class PredicateClassifier implements Serializable {

    private static final long serialVersionUID = -2949247669362202540L;

    private final PerceptronClassifier classifier;
    private final PredicateFeatureGenerator featureGenerator;

    private final static String PREDICATE_LABEL = "predicate";
    private final static String NOT_PREDICATE_LABEL = "not_predicate";

    private final static List<String> LABEL_SET =
            new ImmutableList.Builder<String>().add(
                    NOT_PREDICATE_LABEL, PREDICATE_LABEL).build();

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
     * Returns this classifiers' feature generator.
     *
     * @return feature generator
     */
    public PredicateFeatureGenerator getFeatureGenerator() {
        return featureGenerator;
    }

    /**
     * Returns the set of possible labels for the perceptron.
     *
     * @return set of possible labels--PREDICATE and NOT_PREDICATE
     */
    public static List<String> getLabelSet() {
        return LABEL_SET;
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
            if (PREDICATE_LABEL.equals(predicateClass))
                sentenceAndPredicates.addPredicate(t);
        }

        return sentenceAndPredicates;
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

            String goldLabel = goldSentence.isPredicate(t) ? PREDICATE_LABEL : NOT_PREDICATE_LABEL;
            String predictedLabel = predictedSentence.isPredicate(t) ? PREDICATE_LABEL : NOT_PREDICATE_LABEL;

            BasicDatum<String, String> datum = (BasicDatum<String, String>)
                    featureGenerator.datumFrom(predictedSentence, t);

            datum.setLabel(PerceptronClassifier.formatManualTrainingLabel(predictedLabel, goldLabel));
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

    /**
     * Updates the average weights for this classifier, must be done to
     * classify labels
     */
    public void updateAverageWeights() {
        classifier.updateAverageWeights();
    }
}
