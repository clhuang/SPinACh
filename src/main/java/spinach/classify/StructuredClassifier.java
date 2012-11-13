package spinach.classify;

import spinach.argumentclassifier.ArgumentClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.text.DateFormat;
import java.util.*;

/**
 * A class that does the entire task for a sentence--
 * determines the predicates and the arguments of said sentence.
 * Uses structured learning.
 *
 * @author Calvin Huang
 */
public class StructuredClassifier extends SemanticClassifier {

    private static final int TRAIN_ALL = 0;
    private static final int TRAIN_PREDICATE_C = 1;
    private static final int TRAIN_ARGUMENT_C = 2;
    private transient int trainingMode;

    public boolean VERBOSE = false;

    /**
     * When training argument classifier, use predicted predicates or gold predicates?
     */
    private final boolean PREDICTED_PRED_WHILE_ARG_TRAINING = false;

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * a certain number of epochs and a set of training frames.
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param epochs              number of times to iterate through training datasets
     * @param trainingFrames      set of semantic framesets to train with
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                int epochs, Collection<SemanticFrameSet> trainingFrames) {
        super(argumentClassifier, predicateClassifier, epochs, trainingFrames);
    }

    /**
     * Instantiates a new StructuredClassifier, which runs {@value SemanticClassifier#DEFAULT_EPOCHS} epochs.
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param trainingFrames      set of semantic framesets to train with
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                Collection<SemanticFrameSet> trainingFrames) {
        super(argumentClassifier, predicateClassifier, trainingFrames);
    }

    /**
     * Performs a parse with training weights.
     *
     * @param sentence sentence to parse
     * @return semantic frameset with predicted predicates and arguments
     */
    private SemanticFrameSet trainingParse(TokenSentence sentence) {
        return argumentTrainingParse(predicateTrainingParse(sentence));
    }

    private SemanticFrameSet argumentTrainingParse(TokenSentenceAndPredicates sentence) {
        return argumentClassifier.trainingFramesWithArguments(sentence);
    }

    private TokenSentenceAndPredicates predicateTrainingParse(TokenSentence sentence) {
        return predicateClassifier.trainingSentenceWithPredicates(sentence);
    }

    /**
     * Trains this structured classifier on a collection of known SemanticFrameSets.
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void train(Collection<SemanticFrameSet> goldFrames) {
        setTrainingFrames(goldFrames);
        trainingMode = TRAIN_ALL;
        train();

        argumentClassifier.updateAverageWeights();
        predicateClassifier.updateAverageWeights();
    }

    /**
     * Trains only the argument classifier on a collection of known SemanticFrameSets.
     */
    @Override
    public void trainArgumentClassifier() {
        trainingMode = TRAIN_ARGUMENT_C;
        train();

        argumentClassifier.updateAverageWeights();
    }

    /**
     * Trains only the predicate classifier on a collection of known SemanticFrameSets.
     */
    @Override
    public void trainPredicateClassifier() {
        trainingMode = TRAIN_PREDICATE_C;
        train();

        predicateClassifier.updateAverageWeights();
    }

    private void train() {
        DateFormat df = DateFormat.getDateTimeInstance();

        for (int i = 0; i < epochs; i++) {
            if (VERBOSE) System.out.println();
            System.out.println("Begin training epoch " + (i + 1) + " of " + epochs + " " + df.format(new Date()));

            List<SemanticFrameSet> goldFramesCopy = new ArrayList<SemanticFrameSet>(trainingFrames);

            Collections.shuffle(goldFramesCopy, new Random(i));

            int j = 0;
            for (SemanticFrameSet goldFrame : goldFramesCopy) {
                j++;
                train(goldFrame);
                if (j % 5000 == 0 && VERBOSE)
                    System.out.println("Trained " + j + " sentences of " + trainingFrames.size() + " | " +
                            df.format(new Date()));
            }
        }
    }

    private void train(SemanticFrameSet goldFrame) {

        //when this is run, parse ignores the predicates and semantic data

        switch (trainingMode) {

            case TRAIN_ALL:
                SemanticFrameSet predictedFrame = trainingParse(goldFrame);
                predictedFrame.trimPredicates();

                predicateClassifier.update(predictedFrame, goldFrame);
                argumentClassifier.update(predictedFrame, goldFrame);
                break;

            case TRAIN_ARGUMENT_C:
                predictedFrame = PREDICTED_PRED_WHILE_ARG_TRAINING ?
                        trainingParse(goldFrame) :
                        argumentTrainingParse(goldFrame);
                argumentClassifier.update(predictedFrame, goldFrame);
                break;

            case TRAIN_PREDICATE_C:
                TokenSentenceAndPredicates predictedPredicates = predicateTrainingParse(goldFrame);
                predicateClassifier.update(predictedPredicates, goldFrame);
                break;
        }
    }
}
