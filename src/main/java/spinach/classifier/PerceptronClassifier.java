package spinach.classifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * A classifier based on a multiclass perceptron.
 *
 * @author Calvin Huang
 */
public class PerceptronClassifier implements Classifier, Serializable {

    private static final long serialVersionUID = 1L;
    private static final double ARRAY_INCREMENT_FACTOR = 2;

    /**
     * Each unique feature is assigned a number, as defined in the index.
     * LabelWeights, for each label, provides a vector that when, multiplied
     * by the feature vector for a datum, returns the score for that datum.
     */
    private class LabelWeights implements Serializable {
        private static final long serialVersionUID = 1L;
        private static final int MIN_NUM_FEATURES = 50000;

        /*
        An array of doubles provides the weight for each feature.
        A weighted average of the weights provides a weight used during actual classification.
         */
        private transient double[] weights;
        private transient double[] avgWeights;

        /*
        This keeps track of the number of iterations it has been updated.
        lastUpdateIteration keeps track of the last time the weight for a particular feature
        has been changed, and is used when calculating the average weight for that feature.
         */
        private transient int currentIteration;
        private transient int[] lastUpdateIteration;

        private final String label;

        LabelWeights(int numFeatures, String label) {
            if (numFeatures < MIN_NUM_FEATURES)
                numFeatures = MIN_NUM_FEATURES;
            weights = new double[numFeatures];
            avgWeights = new double[numFeatures];

            /*
            Iteration number is 1-indexed. A 0 in lastUpdateIteration means it hasn't been previously updated.
             */
            lastUpdateIteration = new int[numFeatures];
            currentIteration = 1;

            this.label = label;
        }

        void incrementCurrentIteration() {
            currentIteration++;
        }

        void updateAverage(Set<Integer> exampleFeatureIndices) {
            for (int i : exampleFeatureIndices)
                updateAverageForIndex(i);
        }

        void updateAllAverage() {
            for (int i = 0; i < weights.length; i++)
                updateAverageForIndex(i);
        }

        void updateAverageForIndex(int i) {
            ensureCapacity(i);
            if (lastUpdateIteration[i] != 0)
                avgWeights[i] += weights[i] * (currentIteration - lastUpdateIteration[i]);
            lastUpdateIteration[i] = currentIteration;
        }

        private void ensureCapacity(int index) {
            if (index >= weights.length)
                expand();
        }

        /**
         * Increase array size to accommodate individual features
         */
        private void expand() {
            int newLength = Math.max((int) Math.ceil(weights.length * ARRAY_INCREMENT_FACTOR), featureIndex.size());
            weights = Arrays.copyOf(weights, newLength);
            avgWeights = Arrays.copyOf(avgWeights, newLength);
            lastUpdateIteration = Arrays.copyOf(lastUpdateIteration, newLength);
        }

        void update(Set<Integer> exampleFeatureIndices, double weight) {
            updateAverage(exampleFeatureIndices);

            for (int i : exampleFeatureIndices) {
                ensureCapacity(i);
                weights[i] += weight;
            }
        }

        double trainingDotProduct(Set<Integer> featureCounts) {
            double dotProd = 0;
            for (int i : featureCounts)
                if (i < weights.length)
                    dotProd += weights[i];
            return dotProd;
        }

        double avgDotProduct(Set<Integer> featureCounts) {
            double dotProd = 0;
            for (int i : featureCounts)
                if (i < avgWeights.length)
                    dotProd += avgWeights[i];
            return dotProd;
        }

        private void writeObject(ObjectOutputStream oos) throws IOException {
            oos.defaultWriteObject();
            oos.writeInt(weights.length);
            for (int i = 0; i < weights.length; i++) {
                oos.writeDouble(weights[i]);
                oos.writeDouble(avgWeights[i]);
            }
        }

        private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
            ois.defaultReadObject();
            int length = ois.readInt();

            currentIteration = 1;
            weights = new double[length];
            avgWeights = new double[length];
            lastUpdateIteration = new int[length];

            for (int i = 0; i < length; i++) {
                weights[i] = ois.readDouble();
                avgWeights[i] = ois.readDouble();
                lastUpdateIteration[i] = 1;
            }

        }
    }

    private ArrayList<LabelWeights> zWeights = new ArrayList<LabelWeights>();

    private Index<String> labelIndex = new HashIndex<String>();
    private Index<String> featureIndex = new HashIndex<String>();

    private final int epochs;

    /**
     * Creates a perceptron classifier
     *
     * @param epochs number of times to iterate over dataset
     */
    public PerceptronClassifier(int epochs) {
        this.epochs = epochs;
    }

    /**
     * Creates a perceptron classifier
     *
     * @param initialFeatureSet set of features to start with
     * @param epochs            number of times to iterate over dataset
     */
    public PerceptronClassifier(Set<String> initialFeatureSet, int epochs) {
        this(epochs);
        featureIndex.addAll(initialFeatureSet);
    }

    /**
     * Trains a new classifier based on a dataset
     *
     * @param dataset to be trained on
     */
    @Override
    public void train(Dataset<String, String> dataset) {
        labelIndex = dataset.labelIndex();
        featureIndex = dataset.featureIndex();
        int numFeatures = featureIndex.size();

        zWeights = new ArrayList<LabelWeights>(labelIndex.size());
        for (int i = 0; i < labelIndex.size(); i++)
            zWeights.add(new LabelWeights(numFeatures, labelIndex.get(i)));

        System.err.println("Running perceptronClassifier on " + dataset.size() + " features");
        long startTime = System.currentTimeMillis();

        for (int t = 0; t < epochs; t++) {
            dataset.randomize(t);

            System.err.println();
            System.err.println("Epoch: " + (t + 1) + " of " + epochs);

            for (int i = 0; i < dataset.size(); i++) {
                if (i % 500000 == 0) {
                    System.err.println("Datum: " + i + " of " + dataset.size());
                    System.err.println("Elapsed time: " + (System.currentTimeMillis() - startTime) / 1000 + "s");
                }
                train(dataset.getDatum(i));
            }
        }

        updateAverageWeights();
    }

    /**
     * Trains the classifier based on a dataset with gold/predicted labels for each datum.
     * For use with online learning. Since data in datasets can only contain one label,
     * that label must contain both the predicted and the gold labels.
     *
     * @param goldAndPredictedDataset dataset w/ datum with labels formatted by formatManualTrainingLabel()
     */
    public void manualTrain(Dataset<String, String> goldAndPredictedDataset) {

        for (Datum<String, String> d : goldAndPredictedDataset)
            manualTrain(d);
    }

    private static final char SPACER = Character.LINE_SEPARATOR;

    /**
     * Trains on a single datum with two labels--predicted and gold label
     * Label must be formatted with formatManualTrainingLabel()
     *
     * @param datum datum to train with properly formatted dual label
     */
    private void manualTrain(Datum<String, String> datum) {

        String unformattedLabel = datum.label();
        int separatorPos = unformattedLabel.indexOf(SPACER);

        if (separatorPos < 0)
            throw new IllegalArgumentException("Label is not formatted properly for manual train");

        String predictedLabel = unformattedLabel.substring(0, separatorPos);
        String goldLabel = unformattedLabel.substring(separatorPos + 1);

        train(featuresOf(datum), goldLabel, predictedLabel);
    }

    /**
     * In order to use the manual training functions (where the predicted label has already been determined)
     * the two labels must be formatted to be one label to be put in a datum in a dataset.
     *
     * @param predictedLabel the predicted label
     * @param goldLabel      the gold label
     * @return the two combined labels, in the correct format for manual training
     */
    public static String formatManualTrainingLabel(String predictedLabel, String goldLabel) {
        return predictedLabel + SPACER + goldLabel;
    }

    private void train(Set<Integer> featureIndices, String goldLabel, String predictedLabel) {

        int predictedLabelIndex = labelIndex.indexOf(predictedLabel);
        int goldLabelIndex = labelIndex.indexOf(goldLabel, true);

        if (goldLabelIndex >= zWeights.size())
            zWeights.add(new LabelWeights(featureIndex.size(), goldLabel));

        if (!goldLabel.equals(predictedLabel)) {
            if (predictedLabelIndex >= 0)
                zWeights.get(predictedLabelIndex).update(featureIndices, -1.0);
            zWeights.get(goldLabelIndex).update(featureIndices, 1.0);
        }

        for (LabelWeights zw : zWeights)
            zw.incrementCurrentIteration();
    }

    private void train(Datum<String, String> datum) {
        Set<Integer> exampleFeatureIndices = featuresOf(datum);

        String predictedLabel = argMaxDotProduct(exampleFeatureIndices);
        String goldLabel = datum.label();

        train(exampleFeatureIndices, goldLabel, predictedLabel);
    }

    /*
    Given a datum, returns a set of integers that are the array indices for
    the features in that datum.
     */
    private Set<Integer> featuresOf(Datum<String, String> datum) {
        Set<Integer> featureIndices = new HashSet<Integer>();
        for (String feature : datum.asFeatures()) {
            int index = featureIndex.indexOf(feature, true);
            featureIndices.add(index);
        }
        return featureIndices;
    }

    /**
     * Returns the label that gives the greatest score for some features
     */
    private String argMaxDotProduct(Set<Integer> exampleFeatureIndices) {
        double maxDotProduct = Double.NEGATIVE_INFINITY;
        int argMax = -1;
        for (int i = 0; i < zWeights.size(); i++) {
            double dotProduct = zWeights.get(i).trainingDotProduct(exampleFeatureIndices);
            if (dotProduct > maxDotProduct) {
                maxDotProduct = dotProduct;
                argMax = i;
            }
        }
        if (argMax == -1)
            return null;
        return labelIndex.get(argMax);
    }

    private String argMaxAverageDotProduct(Set<Integer> exampleFeatureIndices) {
        double maxDotProduct = Double.NEGATIVE_INFINITY;
        String argMax = null;
        for (LabelWeights l : zWeights) {
            double dotProduct = l.avgDotProduct(exampleFeatureIndices);
            if (dotProduct > maxDotProduct) {
                maxDotProduct = dotProduct;
                argMax = l.label;
            }
        }
        return argMax;
    }

    /**
     * Returns the scores for each label of some datum
     *
     * @param datum datum to be examined
     * @return Counter with scores of each label
     */
    @Override
    public Counter<String> scoresOf(Datum<String, String> datum) {
        Counter<String> scores = new ClassicCounter<String>();
        Set<Integer> featureCounts = featuresOf(datum);
        for (LabelWeights l : zWeights)
            scores.incrementCount(l.label,
                    l.avgDotProduct(featureCounts));
        return scores;
    }

    /**
     * Returns training scores for each label of some datum
     *
     * @param datum datum to be examined
     * @return Counter with scores of each label
     */
    public Counter<String> trainingScores(Datum<String, String> datum) {
        Counter<String> scores = new ClassicCounter<String>();
        Set<Integer> featureCounts = featuresOf(datum);
        for (int i = 0; i < labelIndex.size(); i++)
            scores.incrementCount(labelIndex.get(i),
                    zWeights.get(i).trainingDotProduct(featureCounts));
        return scores;
    }

    /**
     * Gives the label that is most likely to represent some datum
     *
     * @param datum datum to be examined
     * @return label with highest score
     */
    @Override
    public String classOf(Datum<String, String> datum) {
        return argMaxAverageDotProduct(featuresOf(datum));
    }

    /**
     * Gives the label that is most likely to represent some datum,
     * according to training weights
     *
     * @param datum datum to be examined
     * @return label with highest score
     */
    public String trainingClassOf(Datum<String, String> datum) {
        return argMaxDotProduct(featuresOf(datum));
    }

    /**
     * Clears the weights
     */
    public void reset() {
        zWeights.clear();
    }

    /**
     * Updates all the average weights for accurate results when classifying.
     */
    public void updateAverageWeights() {
        for (LabelWeights l : zWeights)
            l.updateAllAverage();
    }
}
