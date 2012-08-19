package spinach.classify;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.Counter;

import java.io.IOException;

public class StanfordLinearClassifier implements Classifier {

    private LinearClassifier<String, String> linearClassifier;

    public StanfordLinearClassifier(String modelPath){
        linearClassifier = LinearClassifier.readClassifier(modelPath);
    }

    public Counter<String> scoresOf(Datum<String, String> datum) {
        return linearClassifier.scoresOf(datum);
    }

    public String classOf(Datum<String, String> datum){
        return linearClassifier.classOf(datum);
    }

    public void save(String modelPath) throws IOException {
        LinearClassifier.writeClassifier(linearClassifier, modelPath);
    }

    public void train(Dataset<String, String> dataset) {
        linearClassifier =
                new LinearClassifierFactory<String, String>().trainClassifier(dataset);
    }
}
