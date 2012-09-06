package spinach.classify;

import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.featuregen.ExtensibleOnlineFeatureGenerator;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

/**
 * A class that does the entire task for a sentence--
 * determines the predicates and the arguments of said sentence
 *
 * @author Calvin Huang
 */
public class StructuredClassifier implements GEN {

    private final ArgumentClassifier argumentClassifier;
    private final PredicateClassifier predicateClassifier;
    private final int epochs;

    private final double FEATURE_INCREMENT_THRESHOLD = 0.01;

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and a certain number of epochs
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param epochs              number of times to iterate through training datasets
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier, int epochs) {
        this.argumentClassifier = argumentClassifier;
        this.predicateClassifier = predicateClassifier;
        this.epochs = epochs;
    }

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and runs 10 epochs during training
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier) {
        this(argumentClassifier, predicateClassifier, 10);
    }

    /**
     * Given a sentence, determines the predicates and arguments for that sentence
     *
     * @param sentence sentence to analyze
     * @return SemanticFrameSet with the original sentence, along with predicates and arguments
     */
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

    /**
     * Trains this structured classifier on a collection of known SemanticFrameSets
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void train(Collection<SemanticFrameSet> goldFrames) {

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

    /**
     * Trains the argument feature generator for this object's ArgumentClassifier--
     * enables features that increase the F1 of the structured classifier
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void trainArgumentFeatureGenerator(List<SemanticFrameSet> goldFrames) {
        double previousF1;
        Metric metric;

        ExtensibleOnlineFeatureGenerator featureGenerator;
        if (argumentClassifier.isFeatureTrainable())
            featureGenerator = (ExtensibleOnlineFeatureGenerator) argumentClassifier.getFeatureGenerator();
        else {
            System.err.println("Cannot train feature generator--is not a trainable feature generator");
            return;
        }
        featureGenerator.clearFeatures();

        train(goldFrames);
        metric = new Metric(this, goldFrames);
        previousF1 = metric.argumentF1s().getCount(Metric.TOTAL);

        for (int i = 0; i < featureGenerator.numFeatureTypes(); i++) {
            featureGenerator.enableFeatureType(i);
            train(goldFrames);

            metric.recalculateScores();
            double F1 = metric.argumentF1s().getCount(Metric.TOTAL);

            if (F1 > previousF1 + FEATURE_INCREMENT_THRESHOLD)
                previousF1 = F1;
            else
                featureGenerator.disableFeatureType(i);

        }
    }

}
