package spinach.argumentclassifier.featuregen;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

/**
 * A class that, given a SemanticFrameSet and a list of featureTokens,
 * generates a collection of features
 */
public abstract class IndividualFeatureGenerator implements Serializable {
    abstract Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens);

    private String identifier;

    public String getIdentifier() {
        return identifier;
    }

    public boolean equals(Object o) {
        if (o == null)
            return false;
        if (this == o)
            return true;

        if (o instanceof IndividualFeatureGenerator)
            if (this.identifier.equals(((IndividualFeatureGenerator) o).identifier))
                return true;

        return false;

    }

    public int hashCode() {
        return identifier.hashCode();
    }
}
