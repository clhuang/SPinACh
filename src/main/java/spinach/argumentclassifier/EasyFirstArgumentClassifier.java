package spinach.argumentclassifier;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import spinach.classify.Classifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class EasyFirstArgumentClassifier extends ArgumentClassifier{

    public EasyFirstArgumentClassifier(Classifier classifier, ArgumentFeatureGenerator featureGenerator){
        super(classifier, featureGenerator);
    }

    public SemanticFrameSet framesWithArguments(SemanticFrameSet sentenceAndPredicates){

        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates.sentence());
        frameSet.addPredicates(sentenceAndPredicates.getPredicateList());

        for (Token predicate : sentenceAndPredicates.getPredicateList()){
            Map<Token, Counter<String>> argumentLabelScores =
                    new HashMap<Token, Counter<String>>();

            for (Token possibleArg :
                    ArgumentClassifier.argumentCandidates(sentenceAndPredicates, predicate)){
                Counter<String> argClassScores = argClassScores(frameSet, possibleArg, predicate);
                if(!Counters.argmax(argClassScores).equals("NIL"))
                    argumentLabelScores.put(possibleArg, argClassScores);
            }

            while (!argumentLabelScores.isEmpty()){
                Pair<Token, String> argAndLabel = mostCertainArgLabel(argumentLabelScores);
                Token registeredArg = argAndLabel.first();
                String argLabel = argAndLabel.second();

                argumentLabelScores.remove(registeredArg);
                frameSet.addArgument(predicate, registeredArg, argLabel);

                if(!argLabel.equals("NIL") && !argLabel.equals("SU") && !argLabel.startsWith("AM-")){

                    Set<Token> restrictedTokens = frameSet.sentence().getDescendants(registeredArg);
                    restrictedTokens.addAll(frameSet.sentence().getAncestors(registeredArg));

                    for (Token t : argumentLabelScores.keySet()){
                        if (restrictedTokens.contains(t)){
                            Counter<String> tokenLabelScores = argumentLabelScores.get(t);
                            Counter<String> updatedScores = new ClassicCounter<String>();

                            for (Map.Entry<String, Double> e : tokenLabelScores.entrySet()){
                                String label = e.getKey();
                                if (label.equals("NIL") || label.equals("SU") || label.startsWith("AM-"))
                                    updatedScores.setCount(label, e.getValue());
                            }
                        }
                    }
                }

                else if (argLabel.matches("A[0-9]"))
                    for (Token token : argumentLabelScores.keySet()){
                        argumentLabelScores.get(token).remove(argLabel);
                    }
            }

        }

        return frameSet;
    }

    private static Pair<Token, String> mostCertainArgLabel(Map<Token, Counter<String>> argumentLabelScores){
        Pair<Token, String> mostCertainArgLabel = null;
        double highCertainty = Double.NEGATIVE_INFINITY;

        for (Map.Entry<Token, Counter<String>> token : argumentLabelScores.entrySet()){
            String label = Counters.argmax(token.getValue());
            double certainty = token.getValue().getCount(label);
            if (certainty > highCertainty){
                highCertainty = certainty;
                mostCertainArgLabel = new Pair<Token, String>(token.getKey(), label);
            }
        }

        return mostCertainArgLabel;
    }

}
