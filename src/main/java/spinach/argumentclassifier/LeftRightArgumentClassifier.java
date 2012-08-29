package spinach.argumentclassifier;

import edu.stanford.nlp.stats.Counter;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.HashSet;
import java.util.Set;

public class LeftRightArgumentClassifier extends ArgumentClassifier {

    public LeftRightArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
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


    private SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates, boolean training) {
        SemanticFrameSet frameSet = new SemanticFrameSet(sentenceAndPredicates);
        frameSet.addPredicates(sentenceAndPredicates.getPredicateList());

        Set<String> previousLabels = new HashSet<String>();
        Set<Token> restrictedArgs = new HashSet<Token>();

        for (Token predicate : frameSet.getPredicateList()) {

            nextArgument:
            for (Token argument : argumentCandidates(frameSet, predicate)) {

                Counter<String> argClassScores;
                if (training)
                    argClassScores = trainingArgClassScores(frameSet, argument, predicate);
                else
                    argClassScores = argClassScores(frameSet, argument, predicate);

                for (String label : sortArgLabels(argClassScores)) {
                    if (label.equals(ArgumentClassifier.NIL_LABEL)) {
                        continue nextArgument;
                    }

                    if (label.equals("SU") || label.startsWith("AM-")) {
                        frameSet.addArgument(predicate, argument, label);
                        continue nextArgument;
                    }

                    if (previousLabels.contains(label))
                        continue;

                    if (!restrictedArgs.contains(argument)) {
                        restrictedArgs.addAll(frameSet.getAncestors(argument));
                        restrictedArgs.addAll(frameSet.getDescendants(argument));
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
