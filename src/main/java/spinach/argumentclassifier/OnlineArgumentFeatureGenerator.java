package spinach.argumentclassifier;

import edu.stanford.nlp.util.Pair;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.Collection;

public class OnlineArgumentFeatureGenerator extends ArgumentFeatureGenerator{

    @Override
    public Collection<String> featuresOf(SemanticFrameSet sentenceAndPredicates,
                                         Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentenceAndPredicates, argument, predicate);

        for (String argumentRelation : sentenceAndPredicates.argumentsOf(predicate).values())
            features.add("existSemdprel|" + argumentRelation);

        //TODO existCross

        return features;

    }

}
