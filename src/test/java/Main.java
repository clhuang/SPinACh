import edu.stanford.nlp.classify.Dataset;
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

import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args){
        List<SemanticFrameSet> frameSets = CorpusUtils.parseCorpus("src/test/resources/train.closed");

        PerceptronClassifier perceptronClassifier = new PerceptronClassifier();
        ArgumentFeatureGenerator argumentFeatureGenerator = new ArgumentFeatureGenerator();
        ArgumentClassifier argumentClassifier = new EasyFirstArgumentClassifier(perceptronClassifier, argumentFeatureGenerator);

        Dataset<String, String> goldDataset = argumentClassifier.goldDataset(frameSets);



        perceptronClassifier.train(goldDataset);

        perceptronClassifier.save("src/test/resources/argumentClassifier.gz");


        /*PredicateFeatureGenerator predicateFeatureGenerator = new PredicateFeatureGenerator();
        PredicateClassifier predicateClassifier = new PredicateClassifier(new PerceptronClassifier(), predicateFeatureGenerator);
        for (Datum<String, String> d : predicateClassifier.goldDataset(frameSet)){
            System.out.println(d.label());
            System.out.println();
        }*/
    }
}
