package spinach.classify;

import spinach.argumentclassifier.ArgumentClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class StructuredClassifier implements GEN {

    private final ArgumentClassifier argumentClassifier;
    private final PredicateClassifier predicateClassifier;
    private final int epochs;

    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier, int epochs) {
        this.argumentClassifier = argumentClassifier;
        this.predicateClassifier = predicateClassifier;
        this.epochs = epochs;
    }

    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier) {
        this(argumentClassifier, predicateClassifier, 10);
    }

    public ArgumentClassifier getArgumentClassifier() {
        return argumentClassifier;
    }

    public PredicateClassifier getPredicateClassifier() {
        return predicateClassifier;
    }

    public SemanticFrameSet parse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predicateClassifier.sentenceWithPredicates(sentence);

        return argumentClassifier.framesWithArguments(sentenceAndPredicates);

    }

    private SemanticFrameSet trainingParse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predicateClassifier.trainingSentenceWithPredicates(sentence);

        return argumentClassifier.trainingFramesWithArguments(sentenceAndPredicates);

    }

    public void train(List<SemanticFrameSet> goldFrames) {

        for (int i = 0; i < epochs; i++) {

            List<SemanticFrameSet> goldFramesCopy = new ArrayList<SemanticFrameSet>(goldFrames);

            Collections.shuffle(goldFramesCopy, new Random(i));

            for (SemanticFrameSet goldFrame : goldFramesCopy)
                train(goldFrame);
        }
    }

    private void train(SemanticFrameSet goldFrame) {

        //when this is run, parse ignores the predicates and semantic data
        SemanticFrameSet predictedFrame = trainingParse(goldFrame);

        predictedFrame.trimPredicates();

        predicateClassifier.update(predictedFrame, goldFrame);
        argumentClassifier.update(predictedFrame, goldFrame);
    }

}
