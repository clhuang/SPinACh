package spinach.classify;

import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.featuregen.ExtensibleOnlineFeatureGenerator;
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

    public void trainArgumentFeatureGenerator(List<SemanticFrameSet> goldFrames) {
        double previousF1;

        ExtensibleOnlineFeatureGenerator featureGenerator;
        if (argumentClassifier.isFeatureTrainable())
            featureGenerator = (ExtensibleOnlineFeatureGenerator) argumentClassifier.getFeatureGenerator();
        else {
            System.err.println("Cannot train feature generator--is not a trainable feature generator");
            return;
        }
        featureGenerator.clearFeatures();

        train(goldFrames);
        previousF1 = (new Metric(this, goldFrames)).argumentF1s().getCount(Metric.TOTAL);

        for (int i = 0; i < featureGenerator.numFeatureTypes(); i++) {
            featureGenerator.addFeatureType(i);
            train(goldFrames);

            Metric metric = new Metric(this, goldFrames);
            double F1 = metric.argumentF1s().getCount(Metric.TOTAL);

            if (F1 > previousF1)
                previousF1 = F1;
            else
                featureGenerator.removeFeatureType(i);

        }
    }

}
