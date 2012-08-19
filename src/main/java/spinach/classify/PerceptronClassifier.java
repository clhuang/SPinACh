package spinach.classify;

import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;

/**
 * @author Calvin Huang
 *
 * A classifier based on a multiclass perceptron.
 */
public class PerceptronClassifier implements Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    /*
    Each unique feature is assigned a number, as defined in the index.
    LabelWeights, for each label, provides a vector that when, multiplied
    by the feature vector for a datum, returns the score for that datum.
     */
    static class LabelWeights implements Serializable {
        private static final long serialVersionUID = 1L;
        double[] weights;
        int survivalIterations;
        double[] avgWeights;
        private static final double ARRAY_INCREMENT_FACTOR = 2;

        LabelWeights(int numFeatures) {
            weights = new double[numFeatures];
            avgWeights = new double[numFeatures];
            survivalIterations = 0;
        }

        void incrementSurvivalIterations() {
            survivalIterations++;
        }

        private void updateAverage() {
            for(int i = 0; i < weights.length; i++){
                avgWeights[i] += weights[i] * survivalIterations;
            }
        }

        /*
        Increase array size to accomodate individual features
         */
        private void expand(){
            int newLength = (int) Math.ceil(weights.length * ARRAY_INCREMENT_FACTOR);
            weights = Arrays.copyOf(weights, newLength);
            avgWeights = Arrays.copyOf(avgWeights, newLength);
        }

        void update(Set<Integer> exampleFeatureIndices, double weight) {
            updateAverage();

            for(int d : exampleFeatureIndices){
                while (d > weights.length)
                    expand();
                weights[d] += weight;
            }

            survivalIterations = 0;
        }

        double dotProduct(Set<Integer> featureIndices) {
            return dotProduct(featureIndices, weights);
        }

        static double dotProduct(Set<Integer> featureCounts, double [] weights) {
            double dotProd = 0;
            for (int i : featureCounts)
                dotProd += weights[i];
            return dotProd;
        }
    }

    LabelWeights[] zWeights;

    public Index<String> labelIndex;
    public Index<String> featureIndex;

    final int epochs;

    /**
     * Creates a perceptron classifier
     *
     * @param epochs number of times to iterate over dataset
     */
    public PerceptronClassifier(int epochs){
        this.epochs = epochs;
    }

    /**
     * Creates a perceptron classifier that iterates over the data set 10 times
     */
    public PerceptronClassifier(){
        this(10);
    }

    /**
     * Saves the classifier as a gzipped file
     *
     * @param modelPath location to save to
     */
    public void save(String modelPath) {
        try{
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(modelPath))));

            assert(zWeights != null);
            out.writeInt(zWeights.length);
            for(LabelWeights zw: zWeights) {
                out.writeObject(zw);
            }

            out.writeObject(labelIndex);
            out.writeObject(featureIndex);

            out.close();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

    /*
    Loads a classifier from an object stream
     */
    private void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
        int length = in.readInt();
        zWeights = new LabelWeights[length];
        for(int i = 0; i < zWeights.length; i ++){
            zWeights[i] = ErasureUtils.uncheckedCast(in.readObject());
        }

        labelIndex = ErasureUtils.uncheckedCast(in.readObject());
        featureIndex = ErasureUtils.uncheckedCast(in.readObject());
    }

    /**
     * Reads a classifier from a gzip
     *
     * @param modelPath path leading to zipped file
     * @return classifier saved in file
     * @throws ClassNotFoundException
     * @throws IOException
     */
    public static PerceptronClassifier load(String modelPath) throws ClassNotFoundException, IOException {
        GZIPInputStream is = new GZIPInputStream(new FileInputStream(modelPath));

        ObjectInputStream in = new ObjectInputStream(is);
        PerceptronClassifier ex = new PerceptronClassifier();
        ex.load(in);
        in.close();
        is.close();
        return ex;
    }

    /**
     * Trains a new classifier based on a dataset
     *
     * @param dataset to be trained on
     */
    public void train(Dataset<String, String> dataset){
        labelIndex = dataset.labelIndex();
        featureIndex = dataset.featureIndex();
        int numFeatures = featureIndex.size();

        zWeights = new LabelWeights[labelIndex.size()];
        for(int i = 0; i < zWeights.length; i++)
            zWeights[i] = new LabelWeights(numFeatures);

        System.err.println("Running perceptronClassifier on " + dataset.size() + " features");
        long startTime = System.currentTimeMillis();

        for(int t = 0; t < epochs; t++){
            dataset.randomize(t);

            System.err.println();
            System.err.println("Epoch: " + (t+1) + " of " + epochs);

            for(int i = 0; i < dataset.size(); i++){
                if (i%500000 == 0){
                    System.err.println("Datum: " + i + " of " + dataset.size());
                    System.err.println("Elapsed time: " + (System.currentTimeMillis() - startTime)/1000 + "s");
                }
                train(dataset.getDatum(i));
            }

        }
    }

    /**
     * Trains a classifier on a dataset, keeping previous data
     *
     * @param dataset dataset to be trained on
     */
    public void trainIteration(Dataset<String, String> dataset){

        for (int i = 0; i < dataset.size(); i++){
            Datum<String, String> datum = dataset.getDatum(i);

            train(datum);

        }
    }

    private void train(Datum<String, String> datum){
        Set<Integer> exampleFeatureIndices = featuresOf(datum);

        String predictedArg = argMaxDotProduct(exampleFeatureIndices);
        String goldArg = datum.label();

        if (!predictedArg.equals(goldArg)){
            zWeights[labelIndex.indexOf(predictedArg)].update(exampleFeatureIndices, -1.0);
            zWeights[labelIndex.indexOf(goldArg)].update(exampleFeatureIndices, 1.0);
        }

        for(LabelWeights zw : zWeights) {
            zw.incrementSurvivalIterations();
        }
    }

    /*
    Given a datum, returns a set of integers that are the array indices for
    the features in that datum.
     */
    private Set<Integer> featuresOf(Datum<String, String> datum){
        Set<Integer> featureIndices = new HashSet<Integer>();
        for(String feature : datum.asFeatures()){
            int index = featureIndex.indexOf(feature, true);
            featureIndices.add(index);
        }
        return featureIndices;
    }

    /*
    Returns the label that gives the greatest score for some features
     */
    private String argMaxDotProduct(Set<Integer> exampleFeatureIndices){
        double maxDotProduct = Double.NEGATIVE_INFINITY;
        int argMax = -1;
        for(int i = 0; i < zWeights.length; i++){
            double dotProduct = zWeights[i].dotProduct(exampleFeatureIndices);
            if (dotProduct > maxDotProduct){
                maxDotProduct = dotProduct;
                argMax = i;
            }
        }
        return labelIndex.get(argMax);
    }

    private String argMaxAverageDotProduct(Set<Integer> exampleFeatureIndices){
        double maxDotProduct = Double.NEGATIVE_INFINITY;
        int argMax = -1;
        for(int i = 0; i < zWeights.length; i++){
            double dotProduct = LabelWeights.dotProduct(exampleFeatureIndices, zWeights[i].avgWeights);
            if (dotProduct > maxDotProduct){
                maxDotProduct = dotProduct;
                argMax = i;
            }
        }
        return labelIndex.get(argMax);
    }

    /**
     * Returns the scores for each label, of some datum
     *
     * @param datum datum to be examined
     * @return {@link edu.stanford.nlp.stats.Counter Counter} with scores of each label
     */
    public Counter<String> scoresOf(Datum<String, String> datum){
        Counter<String> scores = new ClassicCounter<String>();
        Set<Integer> featureCounts = featuresOf(datum);
        for (int i = 0; i < labelIndex.size(); i++){
            scores.incrementCount(labelIndex.get(i),
                    LabelWeights.dotProduct(featureCounts, zWeights[i].avgWeights));
        }
        return scores;
    }

    private String trainingClassOf(Datum<String, String> datum){
        return argMaxDotProduct(featuresOf(datum));
    }

    /**
     * Gives the label that is most likely to represent some datum
     *
     * @param datum datum to be examined
     * @return label with highest score
     */
    public String classOf(Datum<String, String> datum){
        return argMaxAverageDotProduct(featuresOf(datum));
    }

}
