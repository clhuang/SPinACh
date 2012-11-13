package spinach.argumentclassifier;

import com.google.common.collect.Sets;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class LeftRightArgumentClassifier extends ArgumentClassifier {

    private static final boolean CONSISTENCY_MODULE = false;
    private static final boolean CONS_WHEN_TRAINING = false;

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

            for (Token possibleArg : argumentLabelScores.keySet()) {

                updateCounterScores(frameSet, possibleArg, predicate, argumentLabelScores.get(possibleArg), training);
                String argLabel = Counters.argmax(argumentLabelScores.get(possibleArg));

                if (argLabel != null && !argLabel.equals(NIL_LABEL))
                    frameSet.addArgument(predicate, possibleArg, argLabel);
                else
                    continue;

                if (CONSISTENCY_MODULE && (!training || CONS_WHEN_TRAINING)) {
                    if (argLabel.matches("A[0-9]"))
                        for (Token token : argumentLabelScores.keySet())
                            argumentLabelScores.get(token).remove(argLabel);

                    if (isRestrictedLabel(argLabel)) {

                        Set<Token> restrictedTokens = frameSet.getDescendants(possibleArg);
                        restrictedTokens.addAll(frameSet.getAncestors(possibleArg));

                        for (Token t : Sets.intersection(argumentLabelScores.keySet(), restrictedTokens)) {
                            for (Iterator<String> itr = argumentLabelScores.get(t).keySet().iterator(); itr.hasNext(); )
                                if (isRestrictedLabel(itr.next()))
                                    itr.remove();
                        }
                    }
                }
            }
        }

        return frameSet;
    }

    private boolean isRestrictedLabel(String label) {
        return !(ArgumentClassifier.NIL_LABEL.equals(label) || "SU".equals(label) || label.startsWith("AM-"));
    }
}
