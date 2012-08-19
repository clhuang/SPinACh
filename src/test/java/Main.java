import edu.stanford.nlp.classify.Dataset;
import spinach.CorpusUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.ArgumentFeatureGenerator;
import spinach.argumentclassifier.EasyFirstArgumentClassifier;
import spinach.classify.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;

import java.util.List;

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
