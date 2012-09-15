package spinach.argumentclassifier;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class EasyFirstArgumentClassifier extends ArgumentClassifier {

    public EasyFirstArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
        super(classifier, featureGenerator);
    }

    @Override
    public SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates) {
        return framesWithArguments(sentenceAndPredicates, false);
    }

    @Override
    public SemanticFrameSet trainingFramesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates) {
        return framesWithArguments(sentenceAndPredicates, true);
    }

    public SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates, boolean training) {

        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates);

        for (Token predicate : frameSet.getPredicateList()) {
            Map<Token, Counter<String>> argumentLabelScores =
                    new HashMap<Token, Counter<String>>();

            for (Token possibleArg :
                    ArgumentClassifier.argumentCandidates(sentenceAndPredicates, predicate)) {
                Counter<String> argClassScores;
                if (training)
                    argClassScores = trainingArgClassScores(frameSet, possibleArg, predicate);
                else
                    argClassScores = argClassScores(frameSet, possibleArg, predicate);

                String bestArg = Counters.argmax(argClassScores);
                if (bestArg != null && !bestArg.equals("NIL"))
                    argumentLabelScores.put(possibleArg, argClassScores);
            }

            while (!argumentLabelScores.isEmpty()) {
                Pair<Token, String> argAndLabel = mostCertainArgLabel(argumentLabelScores);
                Token registeredArg = argAndLabel.first();
                String argLabel = argAndLabel.second();

                argumentLabelScores.remove(registeredArg);
                frameSet.addArgument(predicate, registeredArg, argLabel);

                if (!argLabel.equals(ArgumentClassifier.NIL_LABEL) &&
                        !argLabel.equals("SU") && !argLabel.startsWith("AM-")) {

                    Set<Token> restrictedTokens = frameSet.getDescendants(registeredArg);
                    restrictedTokens.addAll(frameSet.getAncestors(registeredArg));

                    for (Token t : argumentLabelScores.keySet()) {
                        if (restrictedTokens.contains(t)) {
                            Counter<String> tokenLabelScores = argumentLabelScores.get(t);
                            Counter<String> updatedScores = new ClassicCounter<String>();

                            for (Map.Entry<String, Double> e : tokenLabelScores.entrySet()) {
                                String label = e.getKey();
                                if (label.equals(ArgumentClassifier.NIL_LABEL)
                                        || label.equals("SU") || label.startsWith("AM-"))
                                    updatedScores.setCount(label, e.getValue());
                            }
                        }
                    }
                } else if (argLabel.matches("A[0-9]"))
                    for (Token token : argumentLabelScores.keySet()) {
                        argumentLabelScores.get(token).remove(argLabel);
                    }
            }

        }

        return frameSet;
    }

    private static Pair<Token, String> mostCertainArgLabel(Map<Token, Counter<String>> argumentLabelScores) {
        Pair<Token, String> mostCertainArgLabel = null;
        double highCertainty = Double.NEGATIVE_INFINITY;

        for (Map.Entry<Token, Counter<String>> token : argumentLabelScores.entrySet()) {
            String label = Counters.argmax(token.getValue());
            double certainty = token.getValue().getCount(label);
            if (certainty > highCertainty) {
                highCertainty = certainty;
                mostCertainArgLabel = new Pair<Token, String>(token.getKey(), label);
            }
        }

        return mostCertainArgLabel;
    }

}
