package spinach.argumentclassifier.featuregen;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.Collection;
import java.util.Collections;

/**
 * IndividualFeatureGenerator that only generates one feature.
 */
public abstract class SingularFeatureGenerator extends IndividualFeatureGenerator {
    /**
     * Instantiates a new SingularFeatureGenerator.
     *
     * @param identifier string that uniquely identifies this IndividualFeatureGenerator
     */
    protected SingularFeatureGenerator(String identifier) {
        super(identifier);
    }

    @Override
    protected final Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
        return Collections.singleton(featureOf(frameSet, predicate, argument));
    }

    /**
     * Generates a feature for a certain sentence, predicate and argument.
     *
     * @param frameSet  sentence with partially parsed semantic data
     * @param predicate predicate of the sentence
     * @param argument  argument of the predicate
     * @return the generated feature
     */
    protected abstract String featureOf(SemanticFrameSet frameSet, Token predicate, Token argument);
}
