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
    final Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument) {
        return Collections.singleton(featureOf(frameSet, predicate, argument));
    }

    abstract String featureOf(SemanticFrameSet frameSet, Token predicate, Token argument);
}
