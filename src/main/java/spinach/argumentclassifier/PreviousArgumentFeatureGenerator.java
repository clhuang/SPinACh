package spinach.argumentclassifier;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.Collection;
import java.util.Map;

public class PreviousArgumentFeatureGenerator extends ArgumentFeatureGenerator{

    @Override
    public Collection<String> featuresOf(SemanticFrameSet sentenceAndPredicates,
                                         Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentenceAndPredicates, argument, predicate);

        Map<Token, String> argumentAndLabels = sentenceAndPredicates.argumentsOf(predicate);

        int mostRecentArgumentIndex = -1;
        String mostRecentLabel = null;

        for (Map.Entry<Token, String> pair : argumentAndLabels.entrySet()){
            int currIndex = pair.getKey().sentenceIndex;
            if (currIndex > mostRecentArgumentIndex){
                mostRecentArgumentIndex = currIndex;
                mostRecentLabel = pair.getValue();
            }
        }

        if (mostRecentLabel == null)
            features.add("previousArgClass:" + ArgumentClassifier.NIL_LABEL);
        else
            features.add("previousArgClass:" + mostRecentLabel);

        return features;
    }

}
