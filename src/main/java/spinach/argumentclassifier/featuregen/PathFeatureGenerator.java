package spinach.argumentclassifier.featuregen;

import java.util.List;

/**
 * A class that generates a feature from a "path" of tokens, or in this instance a list of tokens.
 */
public abstract class PathFeatureGenerator {

    public PathFeatureGenerator(String identifier) {
        this.identifier = identifier;
    }

    abstract String featureOf(List<String> pathTokens);

    String identifier;

    public String getIdentifier() {
        return identifier;
    }
}
