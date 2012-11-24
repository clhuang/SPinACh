package spinach.argumentclassifier.featuregen;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.Serializable;
import java.util.Collection;

/**
 * A class that, given a SemanticFrameSet, a predicate, and an argument,
 * generates a collection of features.
 */
public abstract class IndividualFeatureGenerator implements Serializable {

    /**
     * Instantiates a new IndividualFeatureGenerator.
     *
     * @param identifier string that uniquely identifies this IndividualFeatureGenerator
     */
    protected IndividualFeatureGenerator(String identifier) {
        this.identifier = identifier;
    }

    /**
     * Generates a collection of features for a certain sentence, predicate and argument.
     *
     * @param frameSet  sentence with partially parsed semantic data
     * @param predicate predicate of the sentence
     * @param argument  argument of the predicate
     * @return collection of features
     */
    protected abstract Collection<String> featuresOf(SemanticFrameSet frameSet, Token predicate, Token argument);

    public final String identifier;

    public boolean equals(Object o) {
        if (o == null)
            return false;
        if (this == o)
            return true;

        if (o instanceof IndividualFeatureGenerator)
            if (identifier.equals(((IndividualFeatureGenerator) o).identifier))
                return true;

        return false;

    }

    public int hashCode() {
        return identifier.hashCode();
    }
}
