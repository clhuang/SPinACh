package spinach.argumentclassifier.featuregen;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * A class that, given a SemanticFrameSet and a list of featureTokens,
 * generates a collection of features
 */
public abstract class IndividualFeatureGenerator implements Serializable {

    public IndividualFeatureGenerator(String identifier) {
        this.identifier = identifier;
    }


    abstract Collection<String> featuresOf(SemanticFrameSet frameSet, Map<String, Token> featureTokens);

    String identifier;

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
