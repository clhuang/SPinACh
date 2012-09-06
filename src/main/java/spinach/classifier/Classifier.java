package spinach.classifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.Counter;

/**
 * An interface that allows for classes to utilize a general linear classifier--
 */
public interface Classifier {

    /**
     * Returns the scores for all the possible labels for some datum
     *
     * @param datum datum to be examined
     * @return {@link edu.stanford.nlp.stats.Counter Counter} with possible labels and scores
     */
    public Counter<String> scoresOf(Datum<String, String> datum);

    /**
     * Returns most likely label for some datum
     *
     * @param datum datum to be examined
     * @return most likely label for that datum
     */
    public String classOf(Datum<String, String> datum);

    /**
     * Trains a classifier on some dataset
     *
     * @param dataset to be trained on
     */
    public void train(Dataset<String, String> dataset);
}
