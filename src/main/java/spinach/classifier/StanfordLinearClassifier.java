package spinach.classifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.Counter;

/**
 * Wrapper class for the Stanford LinearClassifier class to implement the Classifier interface
 *
 * @author Calvin Huang
 */
public class StanfordLinearClassifier implements Classifier {

    private LinearClassifier<String, String> linearClassifier;

    public StanfordLinearClassifier(String modelPath) {
        linearClassifier = LinearClassifier.readClassifier(modelPath);
    }

    @Override
    public Counter<String> scoresOf(Datum<String, String> datum) {
        return linearClassifier.scoresOf(datum);
    }

    @Override
    public String classOf(Datum<String, String> datum) {
        return linearClassifier.classOf(datum);
    }

    public void save(String modelPath) {
        LinearClassifier.writeClassifier(linearClassifier, modelPath);
    }

    @Override
    public void train(Dataset<String, String> dataset) {
        linearClassifier =
                new LinearClassifierFactory<String, String>().trainClassifier(dataset);
    }
}
