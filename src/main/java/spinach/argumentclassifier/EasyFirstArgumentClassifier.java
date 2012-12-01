package spinach.argumentclassifier;

import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * An implementation of an ArgumentClassifier that iterates left-to-right
 * through the predicates, and generates scores for each of the predicate's
 * argument candidates simultaneously. It then classifies the argument with
 * the highest score and label, regenerates the scores for the remaining args
 * and repeats until all the arguments have been classified.
 */
public class EasyFirstArgumentClassifier extends ArgumentClassifier {

    private static final long serialVersionUID = 7822422638276122112L;

    private transient Map<Token, Counter<String>> argumentLabelScores;
    private transient SemanticFrameSet frameSet;

    /**
     * Instantiates a new EasyFirstArgumentClassifier.
     *
     * @param classifier       a Perceptron classifier that this ArgumentClassifier is based upon
     * @param featureGenerator that generates features for each input
     */
    public EasyFirstArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
        super(classifier, featureGenerator);
    }

    @Override
    protected SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates, boolean training) {

        frameSet = new SemanticFrameSet(sentenceAndPredicates);

        for (Token predicate : frameSet.getPredicateList()) {
            argumentLabelScores = new LinkedHashMap<Token, Counter<String>>();

            for (Token possibleArg :
                    ArgumentClassifier.argumentCandidates(sentenceAndPredicates, predicate)) {
                argumentLabelScores.put(possibleArg, argClassScores(frameSet, possibleArg, predicate, training));
            }

            while (!argumentLabelScores.isEmpty()) {
                Pair<Token, String> bestArgAndLabel = bestArgAndLabel();
                Token arg = bestArgAndLabel.first();
                String argLabel = bestArgAndLabel.second();

                argumentLabelScores.remove(arg);

                if (argLabel == null || argLabel.equals(NIL_LABEL))
                    continue;

                classifyArg(arg, predicate, argLabel, training);
            }
        }

        return frameSet;
    }

    private Pair<Token, String> bestArgAndLabel() {
        double bestScore = Double.NEGATIVE_INFINITY;
        Token best = null;
        String argMax = null;

        for (Map.Entry<Token, Counter<String>> entry : argumentLabelScores.entrySet()) {
            String bestLabel = Counters.argmax(entry.getValue());
            double value = entry.getValue().getCount(bestLabel);
            if (value > bestScore) {
                bestScore = value;
                best = entry.getKey();
                argMax = bestLabel;
            }
        }

        return new Pair<Token, String>(best, argMax);
    }

    private void classifyArg(Token arg, Token predicate, String argLabel, boolean training) {
        frameSet.addArgument(predicate, arg, argLabel);

        for (Map.Entry<Token, Counter<String>> entry : argumentLabelScores.entrySet())
            updateCounterScores(frameSet, entry.getKey(), predicate, entry.getValue(), training);

        enforceConsistency(predicate, arg, argLabel, frameSet, training, argumentLabelScores);
    }
}
