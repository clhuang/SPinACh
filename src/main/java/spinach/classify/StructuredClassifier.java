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
     * When training argument classifier, use predicted predicates or gold predicates?
     */
    private final boolean PREDICTED_PRED_WHILE_ARG_TRAINING = false;

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
        return argumentTrainingParse(predicateTrainingParse(sentence));
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
                if (j % 5000 == 0)
                    System.out.println("Trained " + j + " sentences of " + trainingFrames.size() + " | " +
                            df.format(new Date()));
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
                predictedFrame = PREDICTED_PRED_WHILE_ARG_TRAINING ?
                        trainingParse(goldFrame) :
                        argumentTrainingParse(goldFrame);
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
     * enables features that increase the F1 of the structured classifier.
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

        //S = {f_0, f_1, ..., f_k}, a random subset of FT
        Random random = new Random(0);

        int numFeaturesToAdd = featureGenerator.featureGeneratorSet().size() / 2;
        Set<IndividualFeatureGenerator> featureGeneratorSet = new HashSet<IndividualFeatureGenerator>();

        Iterator<IndividualFeatureGenerator> it = featureGenerator.featureGeneratorSet().iterator();
        for (int visited = 0; numFeaturesToAdd > 0; visited++) {
            IndividualFeatureGenerator item = it.next();
            if (random.nextDouble() < ((double) numFeaturesToAdd) /
                    (featureGenerator.featureGeneratorSet().size() - visited)) {
                featureGeneratorSet.add(item);
                numFeaturesToAdd--;
            }
        }

        while (true) {

            //C_r = recruitMore(s);
            Set<IndividualFeatureGenerator> additions = recruitMore(featureGeneratorSet);

            //if C_r == {} then return S
            if (additions.isEmpty())
                return;

            //S' = shakeOff(S + C_r)
            Set<IndividualFeatureGenerator> updatedFeatureGeneratorSet = shakeOff(
                    Sets.union(featureGeneratorSet, additions));

            //if scr(M(S)) ≥ scr(M(S′)) then return S
            if (argumentTrainAndScore(featureGeneratorSet) > argumentTrainAndScore(updatedFeatureGeneratorSet))
                return;

            //S = S'
            featureGeneratorSet = updatedFeatureGeneratorSet;
        }
    }

    private double argumentTrainAndScore(Set<IndividualFeatureGenerator> featureGenerators) {
        argumentClassifier.reset();
        featureGenerator.setEnabledFeatureGenerators(featureGenerators);
        trainArgumentClassifier(trainingFrames);
        return new Metric(this, testingFrames).argumentF1s().getCount(
                Metric.TOTAL);
    }

    private Set<IndividualFeatureGenerator> recruitMore(Set<IndividualFeatureGenerator> featureGenerators) {

        //C_r = {}
        Set<IndividualFeatureGenerator> additions = new HashSet<IndividualFeatureGenerator>();

        //p = scr(M(S))
        double originalScore = argumentTrainAndScore(featureGenerators);

        //for each f ∈ FT − S
        for (IndividualFeatureGenerator possibleAddition : featureGenerator.disabledFeatures())
            //if p < scr(M(S + {f})) then C_r += {f};
            if (argumentTrainAndScore(Sets.union(featureGenerators, Collections.singleton(possibleAddition)))
                    > originalScore)
                additions.add(possibleAddition);

        return additions;
    }

    private Set<IndividualFeatureGenerator> shakeOff(Set<IndividualFeatureGenerator> featureGeneratorSet) {

        /*
         * S_max = maxFeatureGenerators
         * S_0 = originalFeatureGenerators
         * S = sortedFeatureGenerators & currentFeatureGenerators
         */

        Set<IndividualFeatureGenerator> maxFeatureGenerators =
                new HashSet<IndividualFeatureGenerator>(featureGeneratorSet);

        while (true) {
            //S_0 = S_max
            Set<IndividualFeatureGenerator> originalFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(maxFeatureGenerators);

            /**
             * Keeps a list of feature generators, and the score attained when
             * the feature generator is not present.
             */
            Map<IndividualFeatureGenerator, Double> featureGenAndScoresWO =
                    new HashMap<IndividualFeatureGenerator, Double>();

            Set<IndividualFeatureGenerator> currentFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(originalFeatureGenerators);

            //track scr(M(S - {f})) for each f ∈ S
            for (IndividualFeatureGenerator generator : originalFeatureGenerators) {
                currentFeatureGenerators.remove(generator);
                featureGenAndScoresWO.put(generator, argumentTrainAndScore(currentFeatureGenerators));
                currentFeatureGenerators.add(generator);
            }

            //sort S in the descending order of scr(M(S − {f})) for each f ∈ S
            Set<IndividualFeatureGenerator> sortedFeatureGenerators =
                    invertedSortByValues(featureGenAndScoresWO).keySet();

            double sMaxScore = argumentTrainAndScore(maxFeatureGenerators);
            ;

            //while (S = S − {f_0}) != {}
            for (; !sortedFeatureGenerators.isEmpty();
                 sortedFeatureGenerators.remove(Iterables.getFirst(sortedFeatureGenerators, null))) {

                //S_max = argmax_(x∈{Smax; S}) scr(M(x))
                double score = argumentTrainAndScore(sortedFeatureGenerators);
                if (score > sMaxScore) {
                    maxFeatureGenerators =
                            new HashSet<IndividualFeatureGenerator>(sortedFeatureGenerators);
                    sMaxScore = score;
                }
            }

            //if S_0 == S_max then return S_0;
            if (originalFeatureGenerators.equals(maxFeatureGenerators))
                return originalFeatureGenerators;
        }
    }

    private static <K, V extends Comparable<V>> Map<K, V> invertedSortByValues(
            final Map<K, V> map) {
        Comparator<K> valueComparator = new Comparator<K>() {
            @Override
            public int compare(K k1, K k2) {
                int compare = map.get(k1).compareTo(map.get(k2));
                return compare == 0 ? 1 : -compare;
            }
        };

        Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll(map);

        return sortedByValues;
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
