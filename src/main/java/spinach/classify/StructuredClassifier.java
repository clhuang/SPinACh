package spinach.classify;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.featuregen.ExtensibleFeatureGenerator;
import spinach.argumentclassifier.featuregen.IndividualFeatureGenerator;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.io.*;
import java.text.DateFormat;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * A class that does the entire task for a sentence--
 * determines the predicates and the arguments of said sentence
 *
 * @author Calvin Huang
 */
public class StructuredClassifier implements GEN {

    private final ArgumentClassifier argumentClassifier;
    private final PredicateClassifier predicateClassifier;
    private final int epochs;

    private transient ExtensibleFeatureGenerator featureGenerator;

    private transient Collection<SemanticFrameSet> trainingFrames;
    private transient Collection<SemanticFrameSet> testingFrames;

    private static final int TRAIN_ALL = 0;
    private static final int TRAIN_PREDICATE_C = 1;
    private static final int TRAIN_ARGUMENT_C = 2;
    private transient int trainingMode;

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and a certain number of epochs
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param epochs              number of times to iterate through training datasets
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier, int epochs) {
        this.argumentClassifier = argumentClassifier;
        this.predicateClassifier = predicateClassifier;
        this.epochs = epochs;
    }

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and runs 10 epochs during training
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier) {
        this(argumentClassifier, predicateClassifier, 10);
    }

    @Override
    public SemanticFrameSet parse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predParse(sentence);

        return argParse(sentenceAndPredicates);

    }

    @Override
    public SemanticFrameSet argParse(TokenSentenceAndPredicates sentence) {
        return argumentClassifier.framesWithArguments(sentence);
    }

    @Override
    public TokenSentenceAndPredicates predParse(TokenSentence sentence) {
        return predicateClassifier.sentenceWithPredicates(sentence);
    }

    /**
     * Performs a parse with training weights.
     *
     * @param sentence sentence to parse
     * @return semantic frameset with predicted predicates and arguments
     */
    private SemanticFrameSet trainingParse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predicateTrainingParse(sentence);

        return argumentTrainingParse(sentenceAndPredicates);

    }

    private SemanticFrameSet argumentTrainingParse(TokenSentenceAndPredicates sentence) {
        return argumentClassifier.trainingFramesWithArguments(sentence);
    }

    private TokenSentenceAndPredicates predicateTrainingParse(TokenSentence sentence) {
        return predicateClassifier.trainingSentenceWithPredicates(sentence);
    }

    /**
     * Trains this structured classifier on a collection of known SemanticFrameSets.
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void train(Collection<SemanticFrameSet> goldFrames) {
        setTrainingFrames(goldFrames);
        trainingMode = TRAIN_ALL;
        train();

        argumentClassifier.updateAverageWeights();
        predicateClassifier.updateAverageWeights();
    }

    /**
     * Trains only the argument classifier on a collection of known SemanticFrameSets.
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void trainArgumentClassifier(Collection<SemanticFrameSet> goldFrames) {
        setTrainingFrames(goldFrames);
        trainingMode = TRAIN_ARGUMENT_C;
        train();

        argumentClassifier.updateAverageWeights();
    }

    /**
     * Trains only the predicate classifier on a collection of known SemanticFrameSets.
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void trainPredicateClassifier(Collection<SemanticFrameSet> goldFrames) {
        setTrainingFrames(goldFrames);
        trainingMode = TRAIN_PREDICATE_C;
        train();

        predicateClassifier.updateAverageWeights();
    }

    private void train() {
        DateFormat df = DateFormat.getDateTimeInstance();
        for (int i = 0; i < epochs; i++) {
            System.out.println();
            System.out.println("Begin training epoch " + (i + 1) + " of " + epochs);

            List<SemanticFrameSet> goldFramesCopy = new ArrayList<SemanticFrameSet>(trainingFrames);

            Collections.shuffle(goldFramesCopy, new Random(i));

            int j = 0;
            for (SemanticFrameSet goldFrame : goldFramesCopy) {
                j++;
                train(goldFrame);
                if (j % 5000 == 0) {
                    Date date = new Date();
                    System.out.println("Trained " + j + " sentences of " + trainingFrames.size() + " | " +
                            df.format(date));
                }
            }
        }
    }

    private void train(SemanticFrameSet goldFrame) {

        //when this is run, parse ignores the predicates and semantic data

        switch (trainingMode) {

            case TRAIN_ALL:
                SemanticFrameSet predictedFrame = trainingParse(goldFrame);
                predictedFrame.trimPredicates();

                predicateClassifier.update(predictedFrame, goldFrame);
                argumentClassifier.update(predictedFrame, goldFrame);
                break;

            case TRAIN_ARGUMENT_C:
                predictedFrame = argumentTrainingParse(goldFrame);
                argumentClassifier.update(predictedFrame, goldFrame);
                break;

            case TRAIN_PREDICATE_C:
                TokenSentenceAndPredicates predictedPredicates = predicateTrainingParse(goldFrame);
                predicateClassifier.update(predictedPredicates, goldFrame);
                break;
        }
    }

    private void setTrainingFrames(Collection<SemanticFrameSet> goldFrames) {
        trainingFrames = goldFrames;
    }

    /**
     * Trains the argument feature generator for this object's ArgumentClassifier--
     * enables features that increase the F1 of the structured classifier
     *
     * @param trainingFrames SemanticFrameSets with known good semantic data, to train on
     * @param testingFrames  SemanticFrameSets with known good semantic data, to test against
     */
    public void trainArgumentFeatureGenerator(List<SemanticFrameSet> trainingFrames,
                                              List<SemanticFrameSet> testingFrames) {
        this.trainingFrames = trainingFrames;
        this.testingFrames = testingFrames;

        if (argumentClassifier.isFeatureTrainable())
            featureGenerator = (ExtensibleFeatureGenerator) argumentClassifier.getFeatureGenerator();
        else {
            System.err.println("Cannot train feature generator--is not a trainable feature generator");
            return;
        }

        featureGenerator.clearFeatures();

        Random random = new Random();

        int numFeaturesToAdd = featureGenerator.featureGeneratorSet().size();
        int visited = 0;
        HashSet<IndividualFeatureGenerator> enabledFeatures = new HashSet<IndividualFeatureGenerator>();

        Iterator<IndividualFeatureGenerator> it = featureGenerator.featureGeneratorSet().iterator();
        while (numFeaturesToAdd > 0) {
            IndividualFeatureGenerator item = it.next();
            if (random.nextDouble() < ((double) numFeaturesToAdd) /
                    (featureGenerator.featureGeneratorSet().size() - visited)) {
                enabledFeatures.add(item);
                numFeaturesToAdd--;
            }
            visited++;
        }
        featureGenerator.setEnabledFeatureGenerators(enabledFeatures);


        Set<IndividualFeatureGenerator> featureGeneratorSet = featureGenerator.enabledFeatures();
        while (true) {
            Set<IndividualFeatureGenerator> additions = recruitMore(featureGeneratorSet);
            if (additions.isEmpty())
                return;

            Set<IndividualFeatureGenerator> updatedFeatureGeneratorSet = shakeOff(
                    Sets.union(featureGeneratorSet, additions));

            if (argumentTrainAndScore(featureGeneratorSet) > argumentTrainAndScore(updatedFeatureGeneratorSet))
                return;

            featureGeneratorSet = updatedFeatureGeneratorSet;
        }
    }

    private double argumentTrainAndScore(Set<IndividualFeatureGenerator> featureGenerators) {
        argumentClassifier.reset();
        predicateClassifier.reset();
        featureGenerator.setEnabledFeatureGenerators(featureGenerators);
        train();
        return new Metric(this, testingFrames).argumentF1s().getCount(
                Metric.TOTAL);
    }

    private Set<IndividualFeatureGenerator> recruitMore(Set<IndividualFeatureGenerator> featureGenerators) {
        Set<IndividualFeatureGenerator> additions = new HashSet<IndividualFeatureGenerator>();
        Set<IndividualFeatureGenerator> possibleAdditions = featureGenerator.disabledFeatures();

        double originalScore = argumentTrainAndScore(featureGenerators);

        for (IndividualFeatureGenerator possibleAddition : possibleAdditions)
            if (argumentTrainAndScore(Sets.union(featureGenerators, Collections.singleton(possibleAddition))) > originalScore)
                additions.add(possibleAddition);

        return additions;
    }

    private Set<IndividualFeatureGenerator> shakeOff(Set<IndividualFeatureGenerator> featureGeneratorSet) {
        Set<IndividualFeatureGenerator> maxFeatureGenerators =
                new HashSet<IndividualFeatureGenerator>(featureGeneratorSet);

        while (true) {
            Set<IndividualFeatureGenerator> originalFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(maxFeatureGenerators);
            Map<IndividualFeatureGenerator, Double> unsortedFeatureGenerators =
                    new HashMap<IndividualFeatureGenerator, Double>();

            Set<IndividualFeatureGenerator> currentFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(originalFeatureGenerators);

            for (IndividualFeatureGenerator generator : originalFeatureGenerators) {
                currentFeatureGenerators.remove(generator);
                unsortedFeatureGenerators.put(generator, argumentTrainAndScore(currentFeatureGenerators));
                currentFeatureGenerators.add(generator);
            }


            double sMaxScore = 0;
            boolean maxScoreUpToDate = false;
            for (Set<IndividualFeatureGenerator> sortedFeatureGenerators =
                         invertedSortByValues(unsortedFeatureGenerators).keySet();
                 !sortedFeatureGenerators.isEmpty();
                 sortedFeatureGenerators.remove(Iterables.getFirst(sortedFeatureGenerators, null))) {

                if (!maxScoreUpToDate) {
                    sMaxScore = argumentTrainAndScore(maxFeatureGenerators);
                    maxScoreUpToDate = true;
                }

                if (argumentTrainAndScore(sortedFeatureGenerators) > sMaxScore) {
                    maxFeatureGenerators =
                            new HashSet<IndividualFeatureGenerator>(sortedFeatureGenerators);
                    maxScoreUpToDate = false;
                }
            }
            if (originalFeatureGenerators.equals(maxFeatureGenerators))
                return originalFeatureGenerators;
        }
    }

    private static <K, V extends Comparable<V>> Map<K, V> invertedSortByValues(final Map<K, V> map) {
        Comparator<K> valueComparator = new Comparator<K>() {
            @Override
            public int compare(K k1, K k2) {
                int compare = map.get(k2).compareTo(map.get(k1));
                if (compare == 0) return 1;
                else return -compare;
            }
        };
        Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll(map);
        return new LinkedHashMap<K, V>(sortedByValues);
    }

    /**
     * Saves this classifier's argument classifier.
     *
     * @param filePath file to save classifier to
     */
    public void exportArgumentClassifier(String filePath) {
        ObjectOutputStream out;

        try {
            out = new ObjectOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(filePath))));

            out.writeObject(argumentClassifier);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads an argument classifier.
     *
     * @param filePath file to load classifier from
     * @return imported classifier
     */
    public static ArgumentClassifier importArgumentClassifier(String filePath) {
        ObjectInputStream in;

        try {
            in = new ObjectInputStream(new BufferedInputStream(
                    new GZIPInputStream(new FileInputStream(filePath))));

            return (ArgumentClassifier) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * Saves this classifier's predicate classifier.
     *
     * @param filePath file to save classifier to
     */
    public void exportPredicateClassifier(String filePath) {
        ObjectOutputStream out;

        try {
            out = new ObjectOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(filePath))));

            out.writeObject(predicateClassifier);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads a predicate classifier.
     *
     * @param filePath file to load classifier from
     * @return imported classifier
     */
    public static PredicateClassifier importPredicateClassifier(String filePath) {
        ObjectInputStream in;

        try {
            in = new ObjectInputStream(new BufferedInputStream(
                    new GZIPInputStream(new FileInputStream(filePath))));

            return (PredicateClassifier) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        return null;
    }
}
