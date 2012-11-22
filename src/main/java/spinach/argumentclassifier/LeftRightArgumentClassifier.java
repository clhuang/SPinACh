package spinach.argumentclassifier;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * An argument classifier implementation that goes from left to right through the predicates, and iterates
 * left to right through each of the predicate's argument candidates, classifying them one by one.
 */
public class LeftRightArgumentClassifier extends ArgumentClassifier {

    private static final long serialVersionUID = 8196815613584637181L;

    /**
     * Instantiates a new LeftRightArgumentClassifier.
     *
     * @param classifier       a Perceptron classifier that this ArgumentClassifier is based upon
     * @param featureGenerator that generates features for each input
     */
    public LeftRightArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
        super(classifier, featureGenerator);
    }

    @Override
    protected SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates, boolean training) {

        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates);

        for (Token predicate : frameSet.getPredicateList()) {
            Map<Token, Counter<String>> argumentLabelScores =
                    new LinkedHashMap<Token, Counter<String>>();

            for (Token possibleArg : argumentCandidates(frameSet, predicate)) {
                Counter<String> argClassScores = new ClassicCounter<String>(classifier.indexedLabels());

                argumentLabelScores.put(possibleArg, argClassScores);
            }

            for (Token arg : argumentLabelScores.keySet()) {

                updateCounterScores(frameSet, arg, predicate, argumentLabelScores.get(arg), training);
                String argLabel = Counters.argmax(argumentLabelScores.get(arg));

                if (argLabel != null && !argLabel.equals(NIL_LABEL)) {
                    frameSet.addArgument(predicate, arg, argLabel);
                    enforceConsistency(predicate, arg, argLabel, frameSet, training, argumentLabelScores);
                }
            }
        }

        return frameSet;
    }
}
