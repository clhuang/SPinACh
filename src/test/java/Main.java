import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.util.Pair;
import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.ArgumentFeatureGenerator;
import spinach.argumentclassifier.EasyFirstArgumentClassifier;
import spinach.classify.PerceptronClassifier;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.predicateclassifier.PredicateFeatureGenerator;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;

import java.util.Map;

public class Main {
    public static void main(String[] args){
        SemanticFrameSet frameSet = CorpusUtils.parseCorpus("src/test/resources/test.closed").get(0);

        TokenSentence sentence = frameSet.sentence();

        /*ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        ArgumentClassifier argumentClassifier = new EasyFirstArgumentClassifier(new PerceptronClassifier(), argumentFeatureGenerator);
        for (Token t : ArgumentClassifier.argumentCandidates(sentence, sentence.tokenAt(3))){
            System.out.println(t.form);
        }*/

        PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();
        PredicateClassifier predicateClassifier = new PredicateClassifier(new PerceptronClassifier(), predicateFeatureGenerator);
        for (Datum<String, String> d : predicateClassifier.goldDataset(frameSet)){
            System.out.println(d.label());
            System.out.println();
        }
    }
}
