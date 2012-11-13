package spinach.argumentclassifier;

import com.google.common.collect.Sets;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class EasyFirstArgumentClassifier extends ArgumentClassifier {

    private static final boolean CONSISTENCY_MODULE = true;
    private static final boolean CONS_WHEN_TRAINING = true;

    public EasyFirstArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
        super(classifier, featureGenerator);
    }

    @Override
    protected SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates, boolean training) {

        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates);

        for (Token predicate : frameSet.getPredicateList()) {
            Map<Token, Counter<String>> argumentLabelScores =
                    new LinkedHashMap<Token, Counter<String>>();

            for (Token possibleArg :
                    ArgumentClassifier.argumentCandidates(sentenceAndPredicates, predicate)) {
                argumentLabelScores.put(possibleArg, argClassScores(frameSet, possibleArg, predicate));
            }

            while (!argumentLabelScores.isEmpty()) {
                Pair<Token, String> bestArgAndLabel = bestArgAndLabel(argumentLabelScores);
                Token arg = bestArgAndLabel.first();
                String argLabel = bestArgAndLabel.second();

                argumentLabelScores.remove(arg);

                if (argLabel == null || argLabel.equals(NIL_LABEL))
                    continue;

                frameSet.addArgument(predicate, arg, argLabel);

                if (CONSISTENCY_MODULE && (!training || CONS_WHEN_TRAINING)) {
                    if (argLabel.matches("A[0-9]"))
                        for (Token token : argumentLabelScores.keySet())
                            argumentLabelScores.get(token).remove(argLabel);
                    else if (isRestrictedLabel(argLabel)) {

                        Set<Token> restrictedTokens = frameSet.getDescendants(arg);
                        restrictedTokens.addAll(frameSet.getAncestors(arg));

                        for (Token t : Sets.intersection(argumentLabelScores.keySet(), restrictedTokens)) {
                            for (Iterator<String> itr = argumentLabelScores.get(t).keySet().iterator(); itr.hasNext(); )
                                if (isRestrictedLabel(itr.next()))
                                    itr.remove();
                        }
                    }
                }

                for (Map.Entry<Token, Counter<String>> entry : argumentLabelScores.entrySet())
                    updateCounterScores(frameSet, entry.getKey(), predicate, entry.getValue(), training);
            }
        }

        return frameSet;
    }

    private static Pair<Token, String> bestArgAndLabel(Map<Token, Counter<String>> argumentLabelScores) {
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

    private boolean isRestrictedLabel(String label) {
        return !(ArgumentClassifier.NIL_LABEL.equals(label) || "SU".equals(label) || label.startsWith("AM-"));
    }
}
