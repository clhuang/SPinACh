package spinach.classify;

import spinach.argumentclassifier.ArgumentClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;

import java.util.Collection;

/**
 * A class that does the entire task for a sentence--
 * determines the predicates and the arguments of said sentence.
 * Does not use structured learning.
 *
 * @author Calvin Huang
 */
public class UnstructuredClassifier extends SemanticClassifier {

    /**
     * Instantiates a new UnstructuredClassifier.
     *
     * @param argumentClassifier  argument classifier to use
     * @param predicateClassifier predicate classifier to use
     * @param epochs              number of times to iterate through training dataset
     * @param trainingFrames      collection of semantic framesets used to train
     */
    public UnstructuredClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                  int epochs, Collection<SemanticFrameSet> trainingFrames) {
        super(argumentClassifier, predicateClassifier, epochs, trainingFrames);
    }

    /**
     * Instantiates a new UnstructuredClassifier, which runs {@value SemanticClassifier#DEFAULT_EPOCHS} epochs.
     *
     * @param argumentClassifier  argument classifier to use
     * @param predicateClassifier predicate classifier to use
     * @param trainingFrames      collection of semantic framesets used to train
     */
    public UnstructuredClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                  Collection<SemanticFrameSet> trainingFrames) {
        super(argumentClassifier, predicateClassifier, trainingFrames);
    }

    @Override
    public void trainArgumentClassifier() {
        argumentClassifier.unstructuredTrain(trainingFrames);
    }

    @Override
    public void trainPredicateClassifier() {
        predicateClassifier.unstructuredTrain(trainingFrames);
    }
}
