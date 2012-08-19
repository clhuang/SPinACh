package spinach.argumentclassifier;

import edu.stanford.nlp.stats.Counter;
import spinach.classify.Classifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;

import java.util.HashSet;
import java.util.Set;

public class LeftRightArgumentClassifier extends ArgumentClassifier{

    public LeftRightArgumentClassifier(Classifier classifier, ArgumentFeatureGenerator featureGenerator) {
        super(classifier, featureGenerator);
    }

    public SemanticFrameSet framesWithArguments(SemanticFrameSet sentenceAndPredicates) {
        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates.sentence());
        frameSet.addPredicates(sentenceAndPredicates.getPredicateList());

        TokenSentence sentence = frameSet.sentence();

        Set<String> previousLabels = new HashSet<String>();
        Set<Token> restrictedArgs = new HashSet<Token>();

        for(Token predicate : frameSet.getPredicateList()){

            nextArgument:
            for (Token argument : argumentCandidates(frameSet, predicate)){

                Counter<String> argClassScores = argClassScores(frameSet, argument, predicate);

                for (String label : sortArgLabels(argClassScores)){
                    if (label.equals("NIL")){
                        continue nextArgument;
                    }

                    if (label.equals("SU") || label.startsWith("AM-")){
                        frameSet.addArgument(predicate, argument, label);
                        continue nextArgument;
                    }

                    if (previousLabels.contains(label))
                        continue;

                    if (!restrictedArgs.contains(argument)){
                        restrictedArgs.addAll(sentence.getAncestors(argument));
                        restrictedArgs.addAll(sentence.getDescendants(argument));
                        frameSet.addArgument(predicate, argument, label);
                        if (label.matches("A[0-9]"))
                            previousLabels.add(label);
                        continue nextArgument;
                    }
                }

            }

        }

        return frameSet;

    }
}
