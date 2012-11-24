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

import java.util.*;

/**
 * A classifier, using Argument and PredicateClassifiers, that classifies semantic roles in a sentence.
 */
public abstract class SemanticClassifier implements GEN {

    protected final ArgumentClassifier argumentClassifier;
    protected final PredicateClassifier predicateClassifier;

    private transient ExtensibleFeatureGenerator featureGenerator;

    protected transient Collection<SemanticFrameSet> trainingFrames;
    protected transient Collection<SemanticFrameSet> testingFrames;
    protected int epochs;

    public static final int DEFAULT_EPOCHS = 10;

    /**
     * Instantiates a new SemanticClassifier.
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param epochs              number of times to iterate through training datasets
     * @param trainingFrames      set of semantic framesets to train with
     */
    protected SemanticClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                 int epochs, Collection<SemanticFrameSet> trainingFrames) {
        this.argumentClassifier = argumentClassifier;
        this.predicateClassifier = predicateClassifier;
        this.trainingFrames = trainingFrames;
        this.epochs = epochs;
    }

    /**
     * Instantiates a new SemanticClassifier, which runs {@value #DEFAULT_EPOCHS} epochs.
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param trainingFrames      set of semantic framesets to train with
     */
    protected SemanticClassifier(ArgumentClassifier argumentClassifier, PredicateClassifier predicateClassifier,
                                 Collection<SemanticFrameSet> trainingFrames) {
        this(argumentClassifier, predicateClassifier, DEFAULT_EPOCHS, trainingFrames);
    }

    @Override
    public SemanticFrameSet parse(TokenSentence sentence) {
        return argParse(predParse(sentence));
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
     * Using the stored training framesets, train the argument classifier.
     */
    public abstract void trainArgumentClassifier();

    /**
     * Using the stored training framesets, train the predicate classifier.
     */
    public abstract void trainPredicateClassifier();

    /**
     * Set the training framesets to use for training classifiers.
     *
     * @param trainingFrames collection of semantic framesets
     */
    public void setTrainingFrames(Collection<SemanticFrameSet> trainingFrames) {
        this.trainingFrames = trainingFrames;
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

        if (!argumentClassifier.isFeatureTrainable()) {
            System.err.println("Cannot train feature generator--is not a trainable feature generator");
            return;
        }
        featureGenerator = (ExtensibleFeatureGenerator) argumentClassifier.getFeatureGenerator();

        featureGenerator.clearFeatures();

        System.err.println("S = {f_0, f_1, ..., f_k}, a random subset of FT");
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

            System.err.println("C_r = recruitMore(s)");
            Set<IndividualFeatureGenerator> additions = recruitMore(featureGeneratorSet);

            System.err.println("if C_r == {} then return S");
            if (additions.isEmpty())
                break;

            System.err.println("S' = shakeOff(S + C_r)");
            Set<IndividualFeatureGenerator> updatedFeatureGeneratorSet = shakeOff(
                    Sets.union(featureGeneratorSet, additions));
            System.err.println("End shakeOff");

            System.err.println("if scr(M(S)) ≥ scr(M(S′)) then return S");
            if (argumentTrainAndScore(featureGeneratorSet) > argumentTrainAndScore(updatedFeatureGeneratorSet))
                break;

            System.err.println("S = S'");
            featureGeneratorSet = updatedFeatureGeneratorSet;
        }

        System.err.print("Final feature generators: ");
        for (IndividualFeatureGenerator f : featureGeneratorSet)
            System.err.print(f.identifier + " ");
        System.err.println();
    }

    private Map<Set<IndividualFeatureGenerator>, Double> calculatedF1s =
            new HashMap<Set<IndividualFeatureGenerator>, Double>();

    private double argumentTrainAndScore(Set<IndividualFeatureGenerator> featureGenerators) {

        System.err.print("Feature generators: ");
        for (IndividualFeatureGenerator f : featureGenerators)
            System.err.print(f.identifier + " ");
        System.err.println();

        if (calculatedF1s.containsKey(featureGenerators)) {
            double f1 = calculatedF1s.get(featureGenerators);
            System.err.println("Previously calculated F1: " + f1);
            return f1;
        }

        argumentClassifier.reset();
        featureGenerator.setEnabledFeatureGenerators(featureGenerators);

        trainArgumentClassifier();

        double f1 = new Metric(this, testingFrames).argumentF1s().getCount(Metric.TOTAL);

        calculatedF1s.put(featureGenerators, f1);
        System.err.println("F1: " + f1);
        System.err.println();

        return f1;
    }

    private Set<IndividualFeatureGenerator> recruitMore(Set<IndividualFeatureGenerator> featureGenerators) {
        System.err.println("Begin recruitMore");

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

        System.err.println("End recruitMore");
        return additions;
    }

    private Set<IndividualFeatureGenerator> shakeOff(Set<IndividualFeatureGenerator> featureGeneratorSet) {
        System.err.println("Begin shakeOff");

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

            System.err.println("track scr(M(S - {f})) for each f ∈ S");
            for (IndividualFeatureGenerator generator : originalFeatureGenerators) {
                currentFeatureGenerators.remove(generator);
                featureGenAndScoresWO.put(generator, argumentTrainAndScore(currentFeatureGenerators));
                currentFeatureGenerators.add(generator);
            }

            System.err.println("sort S in the descending order of scr(M(S − {f})) for each f ∈ S");
            Set<IndividualFeatureGenerator> sortedFeatureGenerators =
                    invertedSortByValues(featureGenAndScoresWO).keySet();

            double sMaxScore = argumentTrainAndScore(maxFeatureGenerators);

            System.err.println("while (S = S − {f_0}) != {}");
            for (; !sortedFeatureGenerators.isEmpty();
                 sortedFeatureGenerators.remove(Iterables.getFirst(sortedFeatureGenerators, null))) {

                System.err.println("S_max = argmax_(x∈{Smax; S}) scr(M(x))");
                double score = argumentTrainAndScore(sortedFeatureGenerators);
                if (score > sMaxScore) {
                    maxFeatureGenerators =
                            new HashSet<IndividualFeatureGenerator>(sortedFeatureGenerators);
                    sMaxScore = score;
                }
            }

            System.err.println("if S_0 == S_max then return S_0");
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

        return new LinkedHashMap<K, V>(sortedByValues);
    }
}
